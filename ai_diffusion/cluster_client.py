from __future__ import annotations
import asyncio
import uuid
from typing import Any, AsyncGenerator, Iterable

from .api import WorkflowInput
from .client import (
    Client,
    ClientEvent,
    ClientFeatures,
    ClientMessage,
    ClientModels,
    DeviceInfo,
    MissingResources,
)
from .comfy_client import ComfyClient
from .settings import PerformanceSettings, settings as _settings
from .util import client_logger as log


class ClusterClient(Client):
    """Client that distributes jobs across multiple ComfyUI backends for parallel generation.

    Implements the Client ABC by wrapping multiple ComfyClient instances.
    Jobs are dispatched to the least-loaded backend. Messages from all backends
    are multiplexed into a single stream with remapped job IDs.
    """

    def __init__(self, backends: list[ComfyClient]):
        self._backends = backends
        # Use first backend's URL as display URL
        self.url = ", ".join(b.url for b in backends)
        # Aggregate device info: sum VRAM, list GPU names
        names = [b.device_info.name for b in backends]
        total_vram = sum(b.device_info.vram for b in backends)
        gpu_type = backends[0].device_info.type if backends else "cuda"
        self.device_info = DeviceInfo(gpu_type, " + ".join(names), total_vram)
        # Use intersection of models across all backends
        self.models = self._compute_model_intersection()
        # Job tracking: local_id -> (backend, backend_job_id)
        self._job_map: dict[str, tuple[ComfyClient, str]] = {}
        # Reverse map: (backend_url, backend_job_id) -> local_id
        self._reverse_map: dict[tuple[str, str], str] = {}
        # Count of in-flight jobs per backend
        self._inflight: dict[str, int] = {b.url: 0 for b in backends}
        # Multiplexed message queue
        self._messages: asyncio.Queue[ClientMessage] = asyncio.Queue()
        # Listener tasks
        self._listener_tasks: list[asyncio.Task] = []
        self._is_connected = False
        # Track dead backends for removal
        self._dead_backends: set[str] = set()

    def _compute_model_intersection(self) -> ClientModels:
        """Compute intersection of models across all backends."""
        if not self._backends:
            return ClientModels()

        base = self._backends[0].models

        if len(self._backends) == 1:
            return base

        # Start with first backend's models, intersect with rest
        result = ClientModels()

        # Checkpoints: intersection by filename
        result.checkpoints = dict(base.checkpoints)
        for backend in self._backends[1:]:
            result.checkpoints = {
                k: v
                for k, v in result.checkpoints.items()
                if k in backend.models.checkpoints
            }

        # VAE: intersection
        result.vae = list(set(base.vae))
        for backend in self._backends[1:]:
            backend_vae = set(backend.models.vae)
            result.vae = [v for v in result.vae if v in backend_vae]

        # LoRAs: intersection
        result.loras = list(set(base.loras))
        for backend in self._backends[1:]:
            backend_loras = set(backend.models.loras)
            result.loras = [l for l in result.loras if l in backend_loras]

        # Upscalers: intersection
        result.upscalers = list(set(base.upscalers))
        for backend in self._backends[1:]:
            backend_upscalers = set(backend.models.upscalers)
            result.upscalers = [u for u in result.upscalers if u in backend_upscalers]

        # Node inputs: use first backend's (should be identical)
        result.node_inputs = base.node_inputs

        # Resources: intersection
        result.resources = dict(base.resources)
        for backend in self._backends[1:]:
            result.resources = {
                k: v
                for k, v in result.resources.items()
                if k in backend.models.resources
            }

        return result

    @staticmethod
    async def connect(urls: list[str], access_token: str = "") -> ClusterClient:
        """Connect to multiple ComfyUI backends and return a ClusterClient."""
        if not urls:
            raise ValueError("No backend URLs provided for cluster mode")

        backends: list[ComfyClient] = []
        errors: list[str] = []

        # Temporarily disable resource checking for individual backends.
        # We check model intersection after all backends connect.
        saved_check = _settings.check_server_resources
        _settings._values["check_server_resources"] = False

        # Connect to all backends in parallel
        async def connect_one(url: str) -> ComfyClient | None:
            try:
                client = await ComfyClient.connect(url)
                log.info(f"Cluster: connected to {url}")
                return client
            except Exception as e:
                log.error(f"Cluster: failed to connect to {url}: {e}")
                errors.append(f"{url}: {e}")
                return None

        try:
            results = await asyncio.gather(*[connect_one(url) for url in urls])
        finally:
            _settings._values["check_server_resources"] = saved_check

        backends = [r for r in results if r is not None]

        if not backends:
            raise ConnectionError(
                f"Failed to connect to any cluster backend:\n"
                + "\n".join(errors)
            )

        if errors:
            log.warning(
                f"Cluster: connected to {len(backends)}/{len(urls)} backends. "
                f"Failed: {', '.join(errors)}"
            )

        client = ClusterClient(backends)

        # Discover models on all backends in parallel
        # Suppress MissingResources -- we handle it at the cluster level
        async def discover_backend(backend: ComfyClient):
            try:
                saved = _settings.check_server_resources
                _settings._values["check_server_resources"] = False
                try:
                    async for _status in backend.discover_models(refresh=False):
                        pass
                finally:
                    _settings._values["check_server_resources"] = saved
            except MissingResources:
                log.warning(f"Cluster: {backend.url} has missing resources, continuing")

        await asyncio.gather(*[discover_backend(b) for b in backends])

        # Recompute model intersection after discovery
        client.models = client._compute_model_intersection()

        # Log model summary
        log.info(
            f"Cluster: {len(backends)} backends, "
            f"{len(client.models.checkpoints)} checkpoints, "
            f"{len(client.models.loras)} loras, "
            f"{client.device_info.vram}GB total VRAM"
        )

        return client

    def _pick_backend(self) -> ComfyClient:
        """Pick the backend with fewest in-flight jobs."""
        alive = [b for b in self._backends if b.url not in self._dead_backends]
        if not alive:
            raise RuntimeError("No healthy backends available in cluster")
        return min(alive, key=lambda b: self._inflight.get(b.url, 0))

    async def enqueue(self, work: WorkflowInput, front: bool = False) -> str:
        local_id = str(uuid.uuid4())
        backend = self._pick_backend()

        backend_id = await backend.enqueue(work, front)

        self._job_map[local_id] = (backend, backend_id)
        self._reverse_map[(backend.url, backend_id)] = local_id
        self._inflight[backend.url] = self._inflight.get(backend.url, 0) + 1

        log.info(
            f"Cluster: job {local_id[:8]} -> {backend.url} "
            f"(backend_id={backend_id[:8]}, inflight={self._inflight[backend.url]})"
        )
        return local_id

    async def _forward_messages(self, backend: ComfyClient):
        """Forward messages from a single backend, remapping job IDs."""
        try:
            async for msg in backend.listen():
                # Remap job_id from backend-local to cluster-local
                if msg.job_id:
                    local_id = self._reverse_map.get((backend.url, msg.job_id))
                    if local_id:
                        msg = ClientMessage(
                            event=msg.event,
                            job_id=local_id,
                            progress=msg.progress,
                            images=msg.images,
                            result=msg.result,
                            error=msg.error,
                        )
                        # Track job completion
                        if msg.event in (
                            ClientEvent.finished,
                            ClientEvent.interrupted,
                            ClientEvent.error,
                        ):
                            self._inflight[backend.url] = max(
                                0, self._inflight.get(backend.url, 1) - 1
                            )
                    else:
                        # No mapping found - could be a status/connected message
                        if msg.event not in (
                            ClientEvent.connected,
                            ClientEvent.disconnected,
                        ):
                            log.warning(
                                f"Cluster: unmapped job_id {msg.job_id} from {backend.url}"
                            )

                # Forward connected/disconnected only for the cluster as a whole
                if msg.event == ClientEvent.connected:
                    # Only forward if this is the first backend connecting
                    continue
                if msg.event == ClientEvent.disconnected:
                    log.warning(f"Cluster: backend {backend.url} disconnected")
                    self._dead_backends.add(backend.url)
                    # Emit errors for any in-flight jobs on this backend
                    for (burl, bid), lid in list(self._reverse_map.items()):
                        if burl == backend.url:
                            await self._messages.put(
                                ClientMessage(
                                    event=ClientEvent.error,
                                    job_id=lid,
                                    error=f"Backend {backend.url} disconnected",
                                )
                            )
                            self._inflight[backend.url] = 0
                    continue

                await self._messages.put(msg)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"Cluster: listener for {backend.url} failed: {e}")
            self._dead_backends.add(backend.url)
            # Emit errors for in-flight jobs
            for (burl, bid), lid in list(self._reverse_map.items()):
                if burl == backend.url:
                    await self._messages.put(
                        ClientMessage(
                            event=ClientEvent.error,
                            job_id=lid,
                            error=f"Backend {backend.url} lost: {e}",
                        )
                    )

    async def listen(self) -> AsyncGenerator[ClientMessage, Any]:
        self._is_connected = True

        # Start a listener task for each backend
        self._listener_tasks = [
            asyncio.get_event_loop().create_task(self._forward_messages(backend))
            for backend in self._backends
        ]

        # Emit initial connected event
        await self._messages.put(ClientMessage(ClientEvent.connected, ""))

        try:
            while self._is_connected:
                yield await self._messages.get()
        except asyncio.CancelledError:
            pass

    async def interrupt(self):
        """Interrupt all backends that are currently executing."""
        tasks = [b.interrupt() for b in self._backends if b.url not in self._dead_backends]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def cancel(self, job_ids: Iterable[str]):
        """Cancel specific jobs, routing to the correct backends."""
        # Group job IDs by backend
        backend_jobs: dict[str, list[str]] = {}
        for local_id in job_ids:
            if local_id in self._job_map:
                backend, backend_id = self._job_map[local_id]
                backend_jobs.setdefault(backend.url, []).append(backend_id)
                self._inflight[backend.url] = max(
                    0, self._inflight.get(backend.url, 1) - 1
                )

        # Cancel on each backend
        tasks = []
        for backend in self._backends:
            if backend.url in self._dead_backends:
                continue
            if backend.url in backend_jobs:
                tasks.append(backend.cancel(backend_jobs[backend.url]))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def disconnect(self):
        """Disconnect from all backends."""
        self._is_connected = False
        for task in self._listener_tasks:
            task.cancel()
        if self._listener_tasks:
            await asyncio.gather(*self._listener_tasks, return_exceptions=True)
        self._listener_tasks = []

        tasks = [b.disconnect() for b in self._backends]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def refresh(self):
        """Refresh models on all backends and recompute intersection."""
        tasks = [b.refresh() for b in self._backends if b.url not in self._dead_backends]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.models = self._compute_model_intersection()

    async def discover_models(self, refresh: bool):
        """Discover models on all backends, yielding aggregate progress."""
        for backend in self._backends:
            if backend.url in self._dead_backends:
                continue
            async for status in backend.discover_models(refresh):
                yield status
        self.models = self._compute_model_intersection()

    @property
    def missing_resources(self):
        # Cluster mode: don't filter styles by missing resources.
        # The fleet may not have every optional resource (inpaint models,
        # upscalers, etc.) but should still allow generation with available
        # checkpoints and LoRAs.
        return None

    @property
    def features(self):
        if self._backends:
            return self._backends[0].features
        return ClientFeatures()

    @property
    def performance_settings(self):
        return PerformanceSettings(
            batch_size=_settings.batch_size,
            resolution_multiplier=_settings.resolution_multiplier,
            max_pixel_count=_settings.max_pixel_count,
            tiled_vae=_settings.tiled_vae,
            dynamic_caching=_settings.dynamic_caching,
        )

    @property
    def backend_count(self) -> int:
        return len(self._backends)

    @property
    def healthy_backend_count(self) -> int:
        return len([b for b in self._backends if b.url not in self._dead_backends])

    def backend_info(self) -> list[dict]:
        """Return info about each backend for UI display."""
        return [
            {
                "url": b.url,
                "device": b.device_info.name,
                "vram": b.device_info.vram,
                "healthy": b.url not in self._dead_backends,
                "inflight": self._inflight.get(b.url, 0),
            }
            for b in self._backends
        ]
