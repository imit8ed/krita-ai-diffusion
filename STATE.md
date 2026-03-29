# Krita AI Diffusion (Cluster Fork) -- Current State

> Last updated: 2026-03-29
> Last session: `krita-cluster-merge-setup`

## Where Things Stand

The `imit8ed/krita-ai-diffusion` fork is merged to upstream 1.49.0 with cluster mode preserved. It is symlinked into Krita's pykrita directory on sparta, with submodules initialized and pycache cleared. The plugin should load on next Krita launch, but cluster mode has not yet been verified against live fleet backends.

### Completed
- Merged upstream/main (1.49.0) into fork -- 20 commits, 2 import conflicts resolved
- Stock plugin backed up to `ai_diffusion.stock-backup` in pykrita dir
- Fork symlinked into `~/Library/Application Support/krita/pykrita/`
- Git submodules (`websockets/`, `debugpy/`) initialized (was causing greyed-out plugin)
- Stale `__pycache__` cleared
- Photoshop alternative research complete (Krita wins)
- Studio integration 3-phase plan drafted

### Not Yet Done
- Verify cluster mode works in Krita (connect to fleet backends)
- Push merge commit to origin (`imit8ed/krita-ai-diffusion`)
- Studio integration Phase 1: Krita export endpoint
- ComfyUI artifact filtering for inpaint debris

## Key Files

| Path | Purpose |
|------|---------|
| `ai_diffusion/cluster_client.py` | ClusterClient -- dispatches to multiple ComfyUI backends |
| `ai_diffusion/connection.py` | Connection layer -- swaps ComfyClient/ClusterClient based on ServerMode |
| `ai_diffusion/ui/settings.py` | Settings UI with cluster mode widget |
| `ai_diffusion/comfy_client.py` | Single-server ComfyClient (same ABC as ClusterClient) |
| `ai_diffusion/websockets/` | Vendored python-websockets (git submodule) |
| `ai_diffusion/debugpy/` | Vendored debugpy (git submodule) |

## How to Resume

```bash
# Check current state
cd ~/dev/third-party/krita-ai-diffusion
git log --oneline -5
git status

# Verify symlinks
ls -la ~/Library/Application\ Support/krita/pykrita/ai_diffusion
ls -la ~/Library/Application\ Support/krita/pykrita/ai_diffusion.desktop

# Push if not already pushed
git push origin main

# Test cluster mode
# 1. Launch Krita
# 2. Settings > Configure Krita > Python Plugin Manager -- "AI Image Diffusion" should be active
# 3. AI Image Generation docker > GPU Cluster button
# 4. Enter fleet URLs: babylon.tadpole-koi.ts.net:8188, atlantis.tadpole-koi.ts.net:8188, olympus.tadpole-koi.ts.net:8188
# 5. Ensure ComfyUI is running on those nodes first (fleet start-service comfyui --host <node>)
```

## Gotchas

- **Submodules required**: After cloning or checking out, always run `git submodule update --init --recursive` -- empty submodule dirs cause the plugin to silently fail to load (greyed out in Plugin Manager)
- **Model intersection**: Cluster mode only shows checkpoints/LoRAs present on ALL backends -- ensure models are synced across fleet nodes
- **Symlink install**: Changes to the fork's working tree are live in Krita immediately (no reinstall needed), but Krita must be relaunched to pick up Python changes

## Plans & Roadmaps

| Location | What |
|----------|------|
| `~/.claude/plans/vectorized-jumping-marble.md` | Krita install plan + Studio integration 3-phase vision |
