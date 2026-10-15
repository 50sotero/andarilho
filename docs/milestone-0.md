# Milestone 0: first real reconstruction

Status, 2026-10-15: **not passed.** The pinned pipeline runs end to end on real
footage and fails honestly where it should, and compression keeps the
appearance. But the only permitted capture so far does not reconstruct into
recognizable rooms along its walking path. Following README section 18, the
application should not expand further until a better capture reconstructs.

## Capture

One permitted handheld phone walkthrough of an apartment, received through
WhatsApp. The media stays out of the repository.

| Property | Value                                                 |
| -------- | ----------------------------------------------------- |
| Video    | H.264, 478 × 850 portrait, 29.9 fps, 70.8 s           |
| Content  | a hallway and rooms, with quick turns in the doorways |
| Scale    | no known dimensions                                   |

WhatsApp re-encodes video. At 478 × 850 the frames carry about a fifth of the
pixels of 1080p, which limits feature matching and splat detail before any
pipeline choice matters.

README asks for three captures with known dimensions: one room, two connected
rooms and a deliberately poor capture. Only this one exists so far.

## Environment

| Item       | Value                                                                                                                         |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------- |
| GPU        | NVIDIA GeForce RTX 5090 Laptop GPU, 24 GB, driver 617.14                                                                      |
| Host       | Windows 11, Docker Desktop (WSL 2 backend)                                                                                    |
| Image      | `infra/Dockerfile.worker-gpu`, base `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@sha256:24c8e358…` (pinned by digest)          |
| Entrypoint | `python -m andarilho_worker.smoke --video … --out … --quality preview`: the worker's own stage code, recipe and normalization |

## Toolchain and licenses

| Component       | Version                      | License                                                                      |
| --------------- | ---------------------------- | ---------------------------------------------------------------------------- |
| COLMAP          | 3.12.5 (built)               | BSD-3-Clause                                                                 |
| nerfstudio      | 1.1.5                        | Apache-2.0                                                                   |
| gsplat          | 1.4.0 (built)                | Apache-2.0                                                                   |
| PyTorch         | 2.7.1+cu128                  | BSD-3-Clause                                                                 |
| torchvision     | 0.22.1                       | BSD-3-Clause                                                                 |
| FFmpeg          | 6.1.1 (Ubuntu 24.04 package) | GPL build (`--enable-gpl`); used only as a separate executable, never linked |
| splat-transform | 1.4.0                        | MIT                                                                          |
| pypdf           | 6.19.0                       | BSD-3-Clause (upload inspection only)                                        |
| CUDA base image | 12.8.1, cuDNN                | NVIDIA Deep Learning Container License                                       |

The recipe's `TOOL_VERSIONS` records FFmpeg as `debian-bookworm`, which is true
of the CPU worker image (where `preparing` runs in production), not of the GPU
image used for these smoke runs.

## Runs

All runs used the `preview` recipe (7000 iterations). Runs 1–7 used the full
video; runs 8–10 used an 8.5 s hallway segment (10.5–19 s), re-encoded without
audio.

| Run | Input   | Outcome                      | Cause and follow-up                                                                                                          |
| --- | ------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| 1   | full    | `CAMERA_REGISTRATION_FAILED` | failed while COLMAP extracted features; the log does not record the underlying error, and later runs extracted features fine |
| 2   | full    | `INSUFFICIENT_OVERLAP`       | the walk splits into several separate COLMAP models (below): an honest failure                                               |
| 3   | full    | `TRAINING_FAILED`            | read-only cache directory in the image; fixed                                                                                |
| 4   | full    | `TRAINING_FAILED`            | Triton/Inductor cache directory not writable; fixed                                                                          |
| 5   | full    | Docker error                 | Docker Desktop lost the container (`unexpected EOF`); environment, not the pipeline                                          |
| 6   | full    | `EXPORT_INVALID`             | `ns-export` could not load the checkpoint under torch 2.7's `weights_only` default; fixed                                    |
| 7   | full    | timeout in `normalizing`     | SOG compression of all spherical-harmonic bands passed 1800 s on the CPU; led to a bounded compression with PLY fallback     |
| 8   | hallway | succeeded (PLY fallback)     | 30 of 34 frames registered, one model; compression hit its 900 s bound, so the 40.8 MB PLY shipped                           |
| 9   | hallway | succeeded (SOG)              | `renderShDegree` 1: SOG in 77 s, 3.1 MB; camera-path frame estimated automatically                                           |
| 10  | hallway | succeeded (SOG)              | rerun of run 9 with GPU memory sampled; this time COLMAP split the 30 registered frames into 2 models                        |

Keyframe sampling experiments on the full video, with the same COLMAP build
(registered frames, and how many separate models they formed):

| Sampling                                      | Frames | Registered | Models | Largest model |
| --------------------------------------------- | ------ | ---------- | ------ | ------------- |
| 2 fps, exhaustive matching                    | 141    | 42         | 4      | 15            |
| 4 fps, sequential matching (overlap 20)       | 283    | 122        | 6      | 33            |
| sharpest of every 3 frames at 12 fps (recipe) | 283    | 155        | 7      | 34            |
| sharpest frames at 12 fps, 400 kept           | 400    | 203        | 8      | 48            |

Better sampling registers more frames, but never into one connected model: the
quick turns between rooms leave no overlap to bridge.

### Resource use (hallway segment)

| Stage                 | Run 8   | Run 9   | Run 10  |
| --------------------- | ------- | ------- | ------- |
| preparing             | 5.3 s   | 5.3 s   | 6.3 s   |
| camera_reconstruction | 13.8 s  | 16.2 s  | 23.6 s  |
| training (7000 it.)   | 306.5 s | 373.4 s | 426.1 s |
| normalizing           | 903.0 s | 77.3 s  | 67.7 s  |

- Splats: 160,000–165,000. Trained PLY: 40.8 MB; SOG 3.0–3.1 MB.
- SOG size and CPU time by kept spherical-harmonic degree: degree 0 in 3 s
  (2.2 MB), degree 1 in 58 s (3.1 MB); degrees 2 and 3 did not finish in 600 s.
- GPU memory (run 10, `nvidia-smi` on the host every 2 s): training held
  0.5–0.9 GB above the desktop's 5.1 GB baseline, at 13–32% utilization. This
  small capture (34 frames at 478 × 850) is far from the GPU's limits; a 1080p
  or 4K re-capture will need fresh measurements.
- Building the GPU image compiles gsplat for about 15 minutes and needs about
  22 GB of RAM; the image is 34.8 GB. Docker Desktop's builder garbage
  collection can evict the 9 GB PyTorch layer between builds (raising its
  `keepStorage` is a Docker Desktop setting on the build machine).

## Findings

**Frame, scale and spawn.** Run 9 estimated the floor from splats under the
camera path: a 12.98° tilt correction, 1,736 inliers, 1.419 m per unit (from a
1.5 m median camera height). Camera heights above that floor ranged from 0.48 m
to 2.91 m, so the bent-reconstruction warning fired. Scale stays `estimated`.
Run 10, on the same segment, estimated a 10.03° tilt and 1.673 m per unit, with
heights from 0.71 m to 2.71 m (warning fired again). The 18% scale difference
between two runs of the same input shows how weakly this capture constrains
the geometry.

**Recognizability.** With the camera-path frame applied to run 8 afterwards (a
prototype of the estimation runs 9 and 10 use), the spawn view showed a recognizable, upright hallway with a doorway and the room beyond.
It degraded into streaks about 2.7 m along the path, where registration thins
out. Run 9's automatic frame faced near-field clutter at spawn and was not
recognizable. Nothing beyond the hallway segment reconstructs, because the
full walk splits into separate models.

**Compression.** Run 9's SOG (spherical harmonics up to degree 1) and its
uncompressed PLY (degree 3), loaded in the viewer from the same manifest and
spawn, were visually indistinguishable apart from faint view-dependent colour.
The SOG is 13 times smaller. Degrees 2 and 3 did not compress within 600 s on
the CPU, and GPU compression needs a Vulkan driver, which neither worker image
has.

**Honest failure.** Once the toolchain worked, the full walkthrough failed with
the user-facing `INSUFFICIENT_OVERLAP` error and no scene, which is the
intended behaviour for a disconnected capture.

## Exit criteria

| Criterion                                                           | Status                                                                                         |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| A single room is recognizable along its walking path                | Not met: no room reconstructs; only the first ~2.7 m of a hallway, and only with run 8's frame |
| A connected two-room capture is recognizable along its walking path | Not met: no such capture yet                                                                   |
| A deliberately poor or disconnected capture fails honestly          | Met: the full video fails with `INSUFFICIENT_OVERLAP` (run 2)                                  |
| Compression preserves acceptable appearance                         | Met for SOG with spherical harmonics up to degree 1                                            |
| Toolchain, licenses and resource use are recorded                   | Met: this document                                                                             |

## Re-capture guidance

- Send the original camera file (a cloud link or cable transfer). Messaging apps
  re-encode video to well under 1080p.
- Record 1080p or 4K at 30 fps with exposure and focus locked, in good, even
  light.
- Walk slowly (about half a metre per second) at a steady chest height, and
  keep the camera pointing along the walk.
- Turn slowly, especially in doorways: a quick turn leaves consecutive frames
  without overlap, which is where this walk split into separate models.
- Walk a loop around each room, then go through the doorway slowly enough that
  both rooms appear in the same frames.
- Measure one reference, such as a door width, so scale can be checked.
- Also record the README's three fixtures: one room, two connected rooms and a
  deliberately poor capture.

## Reproducing

```bash
docker run --rm --gpus all -v "$PWD/capture:/data:ro" -v "$PWD/out:/out" \
  andarilho-worker-gpu:dev \
  python -m andarilho_worker.smoke --video /data/capture.mp4 --out /out --quality preview
```

Open the result in the viewer's dev page:

```bash
VIEWER_BUNDLE_DIR="$PWD/out/bundle" corepack pnpm --filter @andarilho/viewer dev
```

Then visit `/?bundle=/bundle&review=confirm`; `review=confirm` treats the
proposed layout as approved for that dev session only.
