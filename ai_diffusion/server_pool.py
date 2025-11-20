from __future__ import annotations
import asyncio
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from .api import WorkflowInput
from .client import Client, ClientMessage, ClientEvent, DeviceInfo, ClientModels, MissingResources
from .comfy_client import ComfyClient
from .network import NetworkError
from .util import client_logger as log


class ServerStatus(Enum):
    disconnected = 0
    connecting = 1
    connected = 2
    error = 3


@dataclass
class ServerInfo:
    """Information about a server in the pool."""
    url: str
    client: Optional[ComfyClient] = None
    status: ServerStatus = ServerStatus.disconnected
    error: str = ""
    active_jobs: int = 0
    total_jobs: int = 0


class ServerPool:
    """Manages multiple ComfyUI clients for load balancing.

    Distributes jobs across multiple servers using round-robin scheduling.
    Collects results from all servers into a unified message queue.
    """

    def __init__(self, primary_url: str, additional_urls: list[str] = None):
        self.primary_url = primary_url
        self.additional_urls = additional_urls or []
        self.servers: list[ServerInfo] = []
        self._next_server_idx = 0
        self._messages: asyncio.Queue[ClientMessage] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self._is_connected = False

        # Aggregate models from all servers (use primary for now)
        self.models = ClientModels()
        self.device_info: Optional[DeviceInfo] = None
        self.missing_resources: Optional[MissingResources] = None

    @property
    def all_urls(self) -> list[str]:
        """Returns list of all server URLs (primary + additional)."""
        return [self.primary_url] + self.additional_urls

    async def connect(self) -> tuple[ClientModels, DeviceInfo]:
        """Connect to all servers in the pool.

        Returns models and device info from the primary server.
        Additional servers that fail to connect are marked as unavailable.
        """
        if self._is_connected:
            return self.models, self.device_info

        log.info(f"Connecting to server pool with {len(self.all_urls)} server(s)")

        # Initialize server info for all URLs
        self.servers = [ServerInfo(url=url) for url in self.all_urls]

        # Connect to all servers concurrently
        connect_tasks = [self._connect_server(server) for server in self.servers]
        results = await asyncio.gather(*connect_tasks, return_exceptions=True)

        # Check primary server connection (must succeed)
        if isinstance(results[0], Exception):
            raise results[0]

        primary_server = self.servers[0]
        if primary_server.status != ServerStatus.connected or primary_server.client is None:
            raise Exception("Failed to connect to primary server")

        # Use primary server's models and device info
        self.models = primary_server.client.models
        self.device_info = primary_server.client.device_info
        self.missing_resources = primary_server.client.missing_resources

        # Log connection status for all servers
        connected_count = sum(1 for s in self.servers if s.status == ServerStatus.connected)
        log.info(f"Connected to {connected_count}/{len(self.servers)} server(s)")
        for i, server in enumerate(self.servers):
            if server.status == ServerStatus.connected:
                log.info(f"  Server {i+1}: {server.url} - OK")
            else:
                log.warning(f"  Server {i+1}: {server.url} - Failed ({server.error})")

        self._is_connected = True
        return self.models, self.device_info

    async def _connect_server(self, server: ServerInfo):
        """Connect to a single server."""
        try:
            server.status = ServerStatus.connecting
            log.info(f"Connecting to {server.url}")
            server.client = await ComfyClient.connect(server.url)
            server.status = ServerStatus.connected
            log.info(f"Successfully connected to {server.url}")
        except NetworkError as e:
            server.status = ServerStatus.error
            server.error = e.message
            log.error(f"Network error connecting to {server.url}: {e.message}")
            raise
        except Exception as e:
            server.status = ServerStatus.error
            server.error = str(e)
            log.error(f"Failed to connect to {server.url}: {str(e)}")
            raise

    async def discover_models(self, refresh: bool):
        """Discover models on the primary server."""
        primary = self.servers[0]
        if primary.client is None:
            raise Exception("Primary server not connected")

        async for status in primary.client.discover_models(refresh):
            yield status

        self.models = primary.client.models
        self.missing_resources = primary.client.missing_resources

    async def enqueue(self, work: WorkflowInput, front: bool = False) -> str:
        """Enqueue a job to the next available server using round-robin.

        Returns the job ID.
        """
        server = self._get_next_server()
        if server is None or server.client is None:
            raise Exception("No available servers in pool")

        log.info(f"Dispatching job to server: {server.url}")
        job_id = await server.client.enqueue(work, front)
        server.active_jobs += 1
        server.total_jobs += 1
        return job_id

    def _get_next_server(self) -> Optional[ServerInfo]:
        """Get the next available server using round-robin scheduling.

        Skips servers that are not connected.
        """
        available = [s for s in self.servers if s.status == ServerStatus.connected]
        if not available:
            return None

        # Round-robin through available servers
        server = available[self._next_server_idx % len(available)]
        self._next_server_idx += 1
        return server

    async def listen(self):
        """Listen for messages from all servers.

        Yields messages from all servers in the order they arrive.
        """
        if not self._is_connected:
            raise Exception("ServerPool not connected")

        # Start listener tasks for each connected server
        for server in self.servers:
            if server.status == ServerStatus.connected and server.client:
                task = asyncio.create_task(self._listen_server(server))
                self._tasks.append(task)

        # Yield messages as they arrive from any server
        try:
            while self._is_connected:
                msg = await self._messages.get()
                yield msg
        except asyncio.CancelledError:
            pass

    async def _listen_server(self, server: ServerInfo):
        """Listen to messages from a single server."""
        if server.client is None:
            return

        try:
            async for msg in server.client.listen():
                # Decrement active job count when job completes
                if msg.event in (ClientEvent.finished, ClientEvent.error, ClientEvent.interrupted):
                    server.active_jobs = max(0, server.active_jobs - 1)

                # Forward message to unified queue
                await self._messages.put(msg)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"Error listening to server {server.url}: {str(e)}")
            server.status = ServerStatus.error
            server.error = str(e)
            await self._messages.put(ClientMessage(ClientEvent.error, "", error=str(e)))

    async def interrupt(self):
        """Interrupt all running jobs on all servers."""
        tasks = []
        for server in self.servers:
            if server.status == ServerStatus.connected and server.client:
                tasks.append(server.client.interrupt())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def clear_queue(self):
        """Clear job queues on all servers."""
        tasks = []
        for server in self.servers:
            if server.status == ServerStatus.connected and server.client:
                tasks.append(server.client.clear_queue())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def refresh(self):
        """Refresh models on the primary server."""
        primary = self.servers[0] if self.servers else None
        if primary and primary.client:
            await primary.client.refresh()
            self.models = primary.client.models

    async def disconnect(self):
        """Disconnect all servers."""
        if not self._is_connected:
            return

        self._is_connected = False

        # Cancel all listener tasks
        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Disconnect all clients
        disconnect_tasks = []
        for server in self.servers:
            if server.client:
                disconnect_tasks.append(server.client.disconnect())

        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)

        # Reset server states
        for server in self.servers:
            server.client = None
            server.status = ServerStatus.disconnected
            server.active_jobs = 0

        log.info("Disconnected from all servers in pool")

    @property
    def queued_count(self) -> int:
        """Total number of queued jobs across all servers."""
        total = 0
        for server in self.servers:
            if server.status == ServerStatus.connected and server.client:
                total += server.client.queued_count
        return total

    @property
    def is_executing(self) -> bool:
        """Check if any server is executing a job."""
        return any(
            server.status == ServerStatus.connected
            and server.client
            and server.client.is_executing
            for server in self.servers
        )

    @property
    def features(self):
        """Return features from primary server."""
        primary = self.servers[0] if self.servers else None
        if primary and primary.client:
            return primary.client.features
        from .client import ClientFeatures
        return ClientFeatures()

    @property
    def performance_settings(self):
        """Return performance settings from primary server."""
        primary = self.servers[0] if self.servers else None
        if primary and primary.client:
            return primary.client.performance_settings
        from .settings import PerformanceSettings
        return PerformanceSettings()

    def get_pool_stats(self) -> dict:
        """Get statistics about the server pool."""
        return {
            "total_servers": len(self.servers),
            "connected_servers": sum(1 for s in self.servers if s.status == ServerStatus.connected),
            "active_jobs": sum(s.active_jobs for s in self.servers),
            "total_jobs_processed": sum(s.total_jobs for s in self.servers),
            "servers": [
                {
                    "url": s.url,
                    "status": s.status.name,
                    "active_jobs": s.active_jobs,
                    "total_jobs": s.total_jobs,
                    "error": s.error,
                }
                for s in self.servers
            ],
        }
