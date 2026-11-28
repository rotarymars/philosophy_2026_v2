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

## Training

1. Run `scripts/setup.sh` once on the training machine. It uses CUDA automatically if an NVIDIA GPU is visible.
2. Start training from the repository root:
   ```sh
   scripts/train.sh                      # new run in the background, output in models/train.out
   scripts/train.sh --iterations 500     # extra options go to polypack_train
   scripts/train.sh --resume             # continue from models/policy_latest.pt
   tail -f models/train.out              # watch
   pkill -f polypack_train               # stop
   ```
   Every program shows progress bars with elapsed time and an ETA (offline reference, baselines, the whole training run with the current iteration, evaluations); `tail -f models/train.out` shows them as one live line. The script rebuilds first, so pulled code is picked up. Its defaults are `--channels 32 --iterations 200 --eval-every 5`; `./build/polypack_train --help` lists every option. Each minibatch's backward pass runs in pieces of `--micro-batch` samples (default 32): the update is the same, and it fits on small GPUs (a 256-sample minibatch at once needs several GB).

   Training writes these files:
   * `models/policy.pt`: the best model so far, judged by the score on 8 fixed evaluation episodes
   * `models/policy_latest.pt`: the most recent state, including optimizer state, for `--resume`
   * `models/policy_log.csv`: per-iteration metrics
   * `models/train.out`: the console output
3. Evaluate a trained model (GPU if present, all CPU cores):
   ```sh
   scripts/evaluate.sh                   # models/policy.pt on the 8 eval episodes (fast: offline results are cached)
   scripts/evaluate.sh --latest          # models/policy_latest.pt instead
   scripts/evaluate.sh --unseen          # 8 episodes never seen in training; offline reference takes ~2-3 h
                                         # the first time, so it runs in the background (models/eval_unseen.out)
   ```

   Each checkpoint stores its environment and model settings, so it loads on CPU or GPU without extra flags.
3. Evaluate against the baselines and draw the bins:
   ```sh
   ./build/polypack_evaluate --model models/policy.pt --seed-base 42 --episodes 64 --greedy-baseline
   ./build/polypack_evaluate --model models/policy.pt --svg-dir eval_svg
   ```
   Use a `--seed-base` other than the default for an honest test: the default episodes are the ones the trainer picks its best model on. The evaluator also re-checks every packing with the exact overlap test.

### Score

Each episode has a random number of pieces, from 501 to 1000 (`--min-items`, `--items`), with random shapes. The seed picks both the count and the shapes. The player never learns how many pieces are still to come. Every model is evaluated on the same piece range, whatever it was trained with.

For each episode, an offline reference packs the same items while seeing all of them in advance and keeping every bin open. It tries many packing patterns and keeps the fewest bins:
* item orders: largest area first, largest diameter first, largest bounding box first, arrival order, and seeded "noisy area" orders,
* bin choice: first fit or best fit, each piece placed bottom-left.

**score = mean over episodes of (bins used ÷ offline bins)**. Lower is better; 1.0 means as good as the offline reference. The exact optimum is too hard to compute, so the reference is the best packing found: below 1.0 is possible, though rare. When the reference equals ⌈total area⌉ it is proven optimal, and the evaluator reports how often that happens. `--offline-patterns N` sets how many patterns are tried (default 16; it stops early once a packing reaches the lower bound). The offline reference is slow for long episodes (several minutes per pattern for ~750 pieces), so its results are saved in `models/offline_cache.tsv` and reused by later training runs and evaluations with the same settings (`--offline-cache ""` turns this off). Commit that file to share it between machines.

## The learning problem

* **Episode:** a random number (501–1000) of random star-shaped polygons arrive one at a time. Each must be placed immediately and permanently.
* **Bins:** two bins stay open at a time (the bounded-space variant). Opening a third closes the oldest for good.
* **Action:** `(slot, rotation, gx, gy)`, with 3 slots × 8 rotations × 16 × 16 spots = 6,144 actions. Slot 2 means "open a new bin". The 8 rotations are multiples of 90°, plus each of those plus about 45.24° (the exact 5-12-13 rotation). The piece's bounding box starts at `(gx/16, gy/16)`. A spot is legal only if the piece fits there without overlapping anything; every spot is checked exactly each step, with exact shortcuts that skip far-away pieces.
* **What the network sees:** for each (bin, rotation) pair, the bin as a 32×32 picture, a "contact" picture (how snugly the piece would sit at each spot), the exact legality mask, and a thumbnail of the piece. It outputs one score per spot.
* **Reward:** −1 for every bin opened, so the return is −(bins used). The model is scored differently (see [Score](#score)).

## Live viewer

```sh
cmake --build build --target polypack_visualize
python3 viz/server.py        # rebuilds the visualizer, then open http://localhost:8000
```

Pick a player and a seed, then press Start:

* **first fit:** oldest bin, no rotation, lowest and then left-most free spot.
* **greedy (bottom-left fill):** in the oldest bin with room, tries every rotation at its lowest free spot and keeps the one whose top edge ends lowest.
* **trained policy:** a checkpoint from `models/`.

For each piece the viewer first shows, in green, every spot where it fits (checked exactly) and the chosen spot. For a trained policy it also shades each spot by the network's probability. The piece then appears at the chosen spot and slides along its exact path.
