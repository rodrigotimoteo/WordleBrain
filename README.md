# 🧠 WordleBrain

Machine learning system that solves Wordle. Two solvers: an exact **entropy maximizer** (state-of-the-art) and a **trained neural network** that learns to approximate it.

## Architecture

```
src/
  feedback.rs    — Green/Yellow/Grey evaluation with correct duplicate handling
  wordlist.rs    — 14,854-word list loaded at compile time + constraint filtering
  game.rs        — Wordle Game struct (guess tracking, win/finished detection)
  solver.rs      — Solver trait + EntropySolver + RandomSolver + ModelSolver
  model.rs       — Burn neural network (442 → 256 → 128 → 64 → 1)
  train.rs       — Training data generation + manual Adam training loop
  main.rs        — CLI (clap subcommands)
```

## Solvers

### EntropySolver (exact)
Picks the word that maximizes **information gain**:

1. For each candidate guess, simulate feedback against all remaining solutions
2. Group solutions by feedback pattern → partition distribution
3. Pick word with highest entropy: `-Σ(p_i · log₂(p_i))`
4. First turn: 14,854² = ~220M evaluations

**Benchmark**: 100% win rate on word list, ~4.2 avg guesses

### ModelSolver (trained NN)
A feedforward neural network trained to **predict entropy** from game state + candidate word:

- **Input (442)**: 312 state features (green/yellow/grey constraints) + 130 word features (one-hot)
- **Architecture**: 442 → 256 → 128 → 64 → 1 (ReLU activations)
- **Training**: Adam optimizer, MSE loss, supervised on entropy values from EntropySolver
- **Inference**: Scores all remaining words, picks highest — no entropy recomputation needed

### RandomSolver (baseline)
Picks random from remaining candidates. Useful for benchmarking.

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
| `train` | Generate training data + train model (50 games, 100 samples/state, 50 epochs) |
| `train -g 100 -s 200 -e 100` | Custom training params |
| `bench -c 100` | Benchmark entropy solver on 100 random words |
| `bench -c 100 --model` | Benchmark neural network solver |
| `bench -c 100 -r` | Benchmark random solver |

## Training Pipeline

1. **Data generation**: EntropySolver plays N random games. For each game state, sample K random words and compute their true entropy.
2. **Training**: NN learns to map (state, word) → entropy via Adam + MSE.
3. **Saving**: Model saved via Burn's `CompactRecorder` to `artifacts/wordlebrain_model`.

```
cargo run --release -- train -g 50 -s 50 -e 30
```

Training data scales linearly with games × turns × samples/state.

## Backend

Built on [Burn](https://burn.dev) (v0.16), a Rust-native deep learning framework.

**Current**: `ndarray` (CPU).

**GPU acceleration**: Change one line to switch backends:

```rust
// CPU (default)
type Backend = burn::backend::NdArray<f32>;

// GPU via Metal (macOS)
type Backend = burn::backend::Wgpu<f32>;

// GPU via CUDA
type Backend = burn::backend::Cuda<f32>;

// GPU via LibTorch
type Backend = burn::backend::LibTorch<f32>;
```

Add the corresponding feature flag to `Cargo.toml` (`wgpu`, `cuda`, `tch`).

**NPU (Apple Neural Engine)**: Not directly supported. Burn's WGPU backend uses Metal (GPU), not ANE. Apple's ANE requires CoreML — Burn has no CoreML backend.

## Model Persistence

Models are saved to and loaded from disk automatically:

- **Save**: `train::save_model(&model, "artifacts/wordlebrain_model")`
- **Load**: `solver::ModelSolver::from_file("artifacts/wordlebrain_model")`
- **Format**: Burn's `CompactRecorder` binary format

## First Turn Performance

The first turn entropy computation evaluates all 14,854 words against all 14,854 solutions (~220M pattern evaluations). In release mode, this takes ~10-30 seconds on modern CPUs. Subsequent turns are fast (remaining candidates shrinks exponentially).

**Optimization note**: First-turn entropy is state-independent. Could be cached in `artifacts/first_turn_entropy.json` to avoid recomputation.

## Quick Start

```bash
# Build
cargo build --release

# Play interactively
cargo run --release -- play

# Watch AI solve a game
cargo run --release -- ai

# Train a model (takes a few minutes)
cargo run --release -- train -g 50 -s 100 -e 50

# Test the trained model
cargo run --release -- bench -c 20 --model

# Compare solvers
cargo run --release -- bench -c 50          # Entropy
cargo run --release -- bench -c 50 -r       # Random
cargo run --release -- bench -c 50 --model  # Neural network
```
