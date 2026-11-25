# polypack

Online 2D bin packing of random polygons into 1x1 bins, with exact geometry and a reinforcement-learning model trained with libtorch.

* Every coordinate is an exact rational (GMP `mpq_class`), so the geometry has no floating-point error.
* Polygons are written as SVG-like paths, such as `M 0 0 L 1/2 0 L 1/4 1/3 Z` (`M L H V Z`, plus lowercase forms for relative moves).
* Rotations are exact. They're given by `t = tan(θ/2)` for any rational `t`, or as quarter turns.
* `interiorsOverlap(a, b)` answers exactly, and touching edges or corners is allowed.
* **Free placement.** The network only *chooses* an action: a bin, a rotation and a spot on a 16×16 grid. The piece is put directly at that spot, so it can go anywhere that is free, including holes under overhangs. Whether the spot is free is decided exactly. By default the piece then slides down and left from there, exactly as far as it can (`settleBottomLeft`), so pieces end up touching instead of leaving grid-sized gaps (`--settle 0` turns this off).

## Layout

| Path | What |
|---|---|
| `include/polypack`, `src` | geometry library, item generator, exact sliding, environment |
| `train/` | libtorch model, PPO trainer (`polypack_train`), evaluator (`polypack_evaluate`) |
| `models/` | trained checkpoints and training logs (**committed to git**) |
| `tests/` | unit tests, including a randomized cross-check of the overlap test |

## Setup

```sh
scripts/setup.sh
```

The script does five things:
1. Installs missing packages with apt: build-essential, cmake, ninja-build, libgmp-dev, curl and unzip.
2. Detects an NVIDIA GPU.
3. Downloads libtorch and unzips it into `./libtorch` inside the repository. It picks the CUDA 12.4 build when a GPU is visible, and the CPU build otherwise. The folder is about 2 GB and is gitignored.
4. Builds everything into `build/`.
5. Runs the tests.

Options:

| Option | Effect |
|---|---|
| `--variant cpu\|cu118\|cu121\|cu124` | Use this libtorch build instead of choosing automatically |
| `--force-libtorch` | Replace an existing `./libtorch` |
| `--no-install` | Don't install missing packages |
| `--no-tests` | Skip the tests |
| `--debug` | Make a Debug build |

To fetch libtorch on its own, run `scripts/fetch_libtorch.sh [cpu|cu124|...]`. CMake finds `./libtorch` automatically. To use a libtorch somewhere else, pass `-DCMAKE_PREFIX_PATH=/path/to/libtorch`.

A CUDA libtorch also needs the NVIDIA driver and a CUDA toolkit (`nvcc`). The setup script warns if either is missing.

## Training on an NVIDIA GPU

1. Run `scripts/setup.sh` on the GPU machine.
2. Train from the repository root, so that checkpoints land in `models/`:
   ```sh
   ./build/polypack_train --iterations 2000          # uses the GPU automatically
   ./build/polypack_train --help                     # all options
   ./build/polypack_train --resume models/policy_latest.pt --iterations 1000
   ```
   Training writes these files:
   * `models/policy.pt`: the best model so far, judged on 32 fixed evaluation episodes
   * `models/policy_latest.pt`: the most recent state, including optimizer state, for `--resume`
   * `models/policy_log.csv`: per-iteration metrics

   Each checkpoint stores its environment and model settings, so it loads on CPU or GPU without extra flags.
3. Evaluate against the baselines and draw the bins:
   ```sh
   ./build/polypack_evaluate --model models/policy.pt --episodes 64 --svg-dir eval_svg
   ./build/polypack_evaluate --model models/policy.pt --seed-base 42 --greedy-baseline
   ```
   The evaluator also re-checks every packing with the exact overlap test.

Each environment step runs exact geometry on the CPU, spread over `--threads`. The GPU runs the network. A CPU-only libtorch build works too, just slowly.

## The learning problem

* **Episode:** 60 random star-shaped polygons arrive one at a time. Each must be placed immediately and permanently.
* **Bins:** two bins stay open at a time (the bounded-space variant). Opening a third closes the oldest for good.
* **Action:** `(slot, rotation, gx, gy)`, with 3 slots × 8 rotations × 16 × 16 spots = 6,144 actions. Slot 2 means "open a new bin". The 8 rotations are multiples of 90°, plus each of those plus about 45.24° (the exact 5-12-13 rotation). The piece's bounding box starts at `(gx/16, gy/16)`. A spot is legal only if the piece fits there without overlapping anything; every spot is checked exactly each step, with exact shortcuts that skip far-away pieces.
* **What the network sees:** for each (bin, rotation) pair, the bin as a 32×32 picture, a "contact" picture (how snugly the piece would sit at each spot), the exact legality mask, and a thumbnail of the piece. It outputs one score per spot.
* **Reward:** −1 for every bin opened, so the return is −(bins used). The tools also report a lower bound on bins: ⌈total item area⌉.

## Live viewer

```sh
cmake --build build --target polypack_visualize
python3 viz/server.py        # open http://localhost:8000
```

Pick a player and a seed, then press Start:

* **first fit:** oldest bin, no rotation, lowest and then left-most free spot.
* **greedy (bottom-left fill):** in the oldest bin with room, tries every rotation at its lowest free spot and keeps the one whose top edge ends lowest.
* **trained policy:** a checkpoint from `models/`.

For each piece the viewer first shows, in green, every spot where it fits (checked exactly) and the chosen spot. For a trained policy it also shades each spot by the network's probability. The piece then appears at the chosen spot and slides along its exact path.
