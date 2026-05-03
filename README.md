# 🧠 WordleBrain

Machine learning system that solves Wordle. Two solvers: an exact **entropy maximizer** (state-of-the-art) and a **trained neural network** that learns to approximate it. Also available as a **web app** via WASM.

**Try it live**: [rodrigotimoteo.github.io/WordleBrain](https://rodrigotimoteo.github.io/WordleBrain/)

## Workspace Structure

```
wordlebrain/
├── wordlebrain-core/       # Pure Rust lib (no Burn dependency)
│   └── src/                #   feedback, wordlist, game, solver (entropy+random)
├── wordlebrain-wasm/       # WASM crate (wasm-bindgen exports)
│   └── src/lib.rs          #   init, evaluate, solve_full, get_hint, random_word
├── wordlebrain/            # Binary crate (CLI, training, model inference)
│   └── src/                #   main.rs, solver.rs, model.rs, train.rs
├── web/                    # Static site for GitHub Pages
│   ├── index.html          #   Wordle dark theme UI
│   ├── style.css            #   Tiles, keyboard, animations
│   ├── app.js              #   Game logic, WASM calls, UI state
│   └── pkg/                #   WASM output (gitignored, built by wasm-pack)
├── src/words               # Shared 14,854-word dictionary
└── .github/workflows/      # CI: build WASM + deploy to gh-pages
```

### Dependency Graph

```
wordlebrain-core  (pure Rust, no deps beyond std+serde+rand)
    ↑              ↑
wordlebrain    wordlebrain-wasm
(+ burn)       (+ wasm-bindgen)
    ↑
CLI / training
```

## Solvers

### EntropySolver (exact)
Picks the word that maximizes **information gain**:

1. For each candidate guess, simulate feedback against all remaining solutions
2. Group solutions by feedback pattern → partition distribution
3. Pick word with highest entropy: `-Σ(p_i · log₂(p_i))`
4. First turn: 14,854² = ~220M evaluations (cached to disk)

**Benchmark**: 100% win rate, ~4.2 avg guesses

### ModelSolver (trained NN)
A feedforward neural network trained to **predict entropy** from game state + candidate word:

- **Input (442)**: 312 state features (green/yellow/grey constraints) + 130 word features (one-hot)
- **Architecture**: 442 → 256 → 128 → 64 → 1 (ReLU activations)
- **Training**: Adam optimizer, MSE loss, 90/10 train/val split, early stopping (patience=5)
- **Inference**: Scores all remaining words, picks highest — no entropy recomputation needed
- **GPU support**: `--gpu` flag for Metal/CUDA via Burn's WGPU backend

**Benchmark**: 100% win rate, ~4.7 avg guesses

### RandomSolver (baseline)
Picks random from remaining candidates. Useful for benchmarking.

## Web App

Available at [rodrigotimoteo.github.io/WordleBrain](https://rodrigotimoteo.github.io/WordleBrain/).

- **Play tab**: Classic Wordle — type guesses, see colored tiles + keyboard highlighting
- **AI Solve tab**: Watch the entropy solver solve step-by-step or auto-run
- **Stats tab**: Win rate, guess distribution, game history (persisted in localStorage)
- **WASM binary**: ~263KB (words + first-turn entropy cache embedded, zero network fetches)

### Building WASM locally

```bash
wasm-pack build wordlebrain-wasm --target web --out-dir ../web/pkg
cp web/index.html web/style.css web/app.js web/pkg/
python3 -m http.server 8888 --directory web/pkg
# Open http://localhost:8888
```

### WASM API

| Function | Returns |
|----------|---------|
| `init()` | Word count (call first) |
| `evaluate(guess, solution)` | 5-char string: `G`=Green, `Y`=Yellow, `_`=Grey |
| `solve_full(solution)` | JSON array of `{guess, pattern}` steps |
| `solve_step(solution, step)` | JSON: `{guess, pattern, remaining, won}` for one step |
| `get_hint(history_json)` | Best next guess word |
| `random_word()` | Random 5-letter word |
| `validate_word(word)` | Boolean — is it in the dictionary? |
| `word_count()` | Dictionary size |

## CLI

```bash
cargo run --release -- <command>
```

| Command | Description |
|---------|-------------|
| `play` | Interactive Wordle (human guesser) |
| `play --solution crane` | Play with a specific solution |
| `ai` | Watch entropy solver play 1 random game |
| `ai -c 5` | Watch 5 games |
| `ai -r` | Use random solver instead |
| `solve hello` | Entropy solver solves a specific word |
| `solve hello --model` | Neural network solver solves a word |
| `train` | Generate training data + train model (default: 50 games, 100 samples, 50 epochs) |
| `train -g 100 -s 200 -e 100 --gpu` | Train with GPU acceleration |
| `bench -c 100` | Benchmark entropy solver on 100 random words |
| `bench -c 100 --model` | Benchmark neural network solver |
| `bench -c 100 -r` | Benchmark random solver |

## Training Pipeline

1. **Data generation**: EntropySolver plays N random games. For each game state, sample K random words and compute their true entropy.
2. **Training**: NN learns to map (state, word) → entropy via Adam + MSE.
3. **90/10 split**: 90% training, 10% validation with early stopping (patience=5).
4. **Saving**: Model saved via Burn's `CompactRecorder` to `artifacts/wordlebrain_model`.

```bash
# CPU training
cargo run --release -- train -g 50 -s 100 -e 50

# GPU training (Metal on macOS)
cargo run --release --features wgpu -- train -g 50 -s 100 -e 50 --gpu
```

## Backend

Built on [Burn](https://burn.dev) (v0.16), a Rust-native deep learning framework.

| Backend | Feature flag | Command |
|---------|-------------|---------|
| CPU (default) | `ndarray` | `cargo run --release -- train` |
| GPU (Metal/CUDA/Vulkan) | `wgpu` | `cargo run --release --features wgpu -- train --gpu` |

**NPU (Apple Neural Engine)**: Not supported. Burn's WGPU backend uses Metal (GPU), not ANE.

## Model Persistence

- **Save**: `train::save_model(&model, "artifacts/wordlebrain_model")`
- **Load**: `solver::ModelSolver::from_file("artifacts/wordlebrain_model")`
- **Format**: Burn's `CompactRecorder` binary

## First Turn Performance

The first turn entropy computation evaluates all 14,854 words against all 14,854 solutions (~220M pattern evaluations). This is cached to `artifacts/first_turn_cache.json` — computed once (~22s), then loaded instantly on subsequent runs.

## GitHub Pages Deployment

Pushing to `master` triggers the GitHub Actions workflow (`.github/workflows/deploy.yml`):

1. Builds WASM with `wasm-pack`
2. Copies `web/index.html`, `web/style.css`, `web/app.js` into `web/pkg/`
3. Deploys `web/pkg/` to the `gh-pages` branch

**Setup**: Go to Settings → Pages → Source: select "Deploy from a branch" → Branch: `gh-pages` → Folder: `/(root)`.

## Quick Start

```bash
# Build CLI
cargo build --release

# Play interactively
cargo run --release -- play

# Watch AI solve a game
cargo run --release -- ai

# Train a model
cargo run --release -- train -g 50 -s 100 -e 50

# Benchmark solvers
cargo run --release -- bench -c 20          # Entropy
cargo run --release -- bench -c 20 --model  # Neural network

# Build web app locally
wasm-pack build wordlebrain-wasm --target web --out-dir ../web/pkg
cp web/index.html web/style.css web/app.js web/pkg/
python3 -m http.server 8888 --directory web/pkg
# Open http://localhost:8888
```