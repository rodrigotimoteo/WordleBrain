mod model;
mod solver;
mod train;

use clap::{Parser, Subcommand};
use rand::seq::SliceRandom;
use std::io::{self, Write};

use wordlebrain_core::feedback::Feedback;
use wordlebrain_core::game::Game;
use crate::solver::{EntropySolver, ModelSolver, RandomSolver, Solver};

#[derive(Parser)]
#[command(name = "wordlebrain", about = "Machine learning Wordle solver")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Play Wordle interactively (human guesser)
    Play {
        /// Use a specific solution word (random if not provided)
        #[arg(short, long)]
        solution: Option<String>,
    },
    /// Watch an AI solver play random games
    Ai {
        /// Number of games to watch
        #[arg(short, long, default_value = "1")]
        count: usize,
        /// Use random solver instead of entropy solver
        #[arg(short, long)]
        random: bool,
    },
    /// Solve a specific word using a solver
    Solve {
        /// The word to solve
        word: String,
        /// Use the trained neural network model instead of entropy solver
        #[arg(short, long)]
        model: bool,
        /// Path to the trained model file
        #[arg(long, default_value = "artifacts/wordlebrain_model")]
        model_path: String,
    },
    /// Generate training data and train the neural network model
    Train {
        /// Number of games to generate training data from
        #[arg(short, long, default_value = "50")]
        games: usize,
        /// Number of word samples per game state
        #[arg(short, long, default_value = "100")]
        samples: usize,
        /// Number of training epochs
        #[arg(short, long, default_value = "50")]
        epochs: usize,
        /// Batch size
        #[arg(short, long, default_value = "256")]
        batch: usize,
        /// Learning rate
        #[arg(long, default_value = "0.001")]
        lr: f64,
        /// Use GPU (WGPU/Metal) backend instead of CPU
        #[arg(long)]
        gpu: bool,
    },
    /// Benchmark solver performance on random words
    Bench {
        /// Number of games to benchmark
        #[arg(short, long, default_value = "100")]
        count: usize,
        /// Use random solver instead of entropy solver
        #[arg(short, long)]
        random: bool,
        /// Use the trained neural network model instead of entropy solver
        #[arg(short, long)]
        model: bool,
        /// Path to the trained model file
        #[arg(long, default_value = "artifacts/wordlebrain_model")]
        model_path: String,
    },
}

fn main() {
    let cli = Cli::parse();
    let all_words = wordlebrain_core::wordlist::load_words();

    match cli.command {
        Command::Play { solution } => cmd_play(&all_words, solution),
        Command::Ai { count, random } => cmd_ai(&all_words, count, random),
        Command::Solve { word, model, model_path } => cmd_solve(&all_words, &word, model, &model_path),
        Command::Train {
            games,
            samples,
            epochs,
            batch,
            lr,
            gpu,
        } => cmd_train(&all_words, games, samples, epochs, batch, lr, gpu),
        Command::Bench { count, random, model, model_path } => cmd_bench(&all_words, count, random, model, &model_path),
    }
}

// ── Play ────────────────────────────────────────────────────────────────────

fn cmd_play(all_words: &[String], solution: Option<String>) {
    let solution = solution.unwrap_or_else(|| {
        all_words
            .choose(&mut rand::thread_rng())
            .cloned()
            .unwrap()
    });
    let mut game = Game::new(solution.clone());

    println!("\n🎮 WordleBrain — Interactive Play");
    println!("   Guess the 5-letter word in 6 tries.");
    println!("   🟩 = correct letter, correct position");
    println!("   🟨 = correct letter, wrong position");
    println!("   ⬛ = letter not in word\n");

    loop {
        print!("Guess {}> ", game.turns_used() + 1);
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let word = input.trim().to_lowercase();

        if word.len() != 5 || !word.chars().all(|c| c.is_ascii_lowercase()) {
            println!("   Enter exactly 5 letters (a-z).");
            continue;
        }

        let pattern = game.guess(&word);
        print_pattern(&pattern, &word);

        if game.is_won() {
            println!("🎉 Solved in {} guesses!", game.turns_used());
            break;
        }
        if game.is_finished() {
            println!("💀 Out of guesses! Word was: {}", solution);
            break;
        }
    }
}

// ── AI ──────────────────────────────────────────────────────────────────────

fn cmd_ai(all_words: &[String], count: usize, random: bool) {
    for g in 0..count {
        let solution = all_words
            .choose(&mut rand::thread_rng())
            .cloned()
            .unwrap();
        let mut game = Game::new(solution.clone());
        let mut remaining = all_words.to_vec();

        let solver: &dyn Solver = if random {
            &RandomSolver
        } else {
            &EntropySolver
        };

        println!("\n🤖 Game {} — Solution: {}", g + 1, solution);

        loop {
            let guess = solver.next_guess(&remaining, all_words, game.history());
            let pattern = game.guess(&guess);
            println!(
                "   Turn {}: {:<5}  {}",
                game.turns_used(),
                guess,
                pattern_string(&pattern)
            );

            if game.is_won() {
                println!("   ✅ Solved in {}!", game.turns_used());
                break;
            }
            if game.is_finished() {
                println!("   ❌ Failed! Word: {}", solution);
                break;
            }
            remaining = wordlebrain_core::wordlist::filter(&remaining, game.history());
        }
    }
}

// ── Solve ───────────────────────────────────────────────────────────────────

fn cmd_solve(all_words: &[String], word: &str, use_model: bool, model_path: &str) {
    let word = word.to_lowercase();
    if word.len() != 5 || !all_words.contains(&word) {
        println!("'{}' is not a valid 5-letter word.", word);
        return;
    }

    let mut game = Game::new(word.clone());
    let mut remaining = all_words.to_vec();

    println!("\n🧠 Solving: {}", word);
    if use_model {
        println!("   Using: trained neural network model ({})", model_path);
    }

    let model_solver: Option<ModelSolver> = if use_model {
        Some(ModelSolver::from_file(model_path))
    } else {
        None
    };

    loop {
        let guess = if let Some(ref ms) = model_solver {
            ms.next_guess(&remaining, all_words, game.history())
        } else {
            let solver = EntropySolver;
            solver.next_guess(&remaining, all_words, game.history())
        };
        let pattern = game.guess(&guess);
        println!(
            "   Turn {}: {:<5}  {}",
            game.turns_used(),
            guess,
            pattern_string(&pattern)
        );

        if game.is_won() {
            println!("   ✅ Solved in {}!", game.turns_used());
            break;
        }
        if game.is_finished() {
            println!("   ❌ Failed!");
            break;
        }
        remaining = wordlebrain_core::wordlist::filter(&remaining, game.history());
    }
}

// ── Train ───────────────────────────────────────────────────────────────────

fn cmd_train(
    all_words: &[String],
    num_games: usize,
    samples_per_state: usize,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    use_gpu: bool,
) {
    println!("\n🧠 Training WordleBrain neural network");
    println!(
        "   Games: {}, samples/state: {}, epochs: {}, batch: {}, lr: {}",
        num_games, samples_per_state, epochs, batch_size, lr
    );

    // Generate training data
    println!("\n📊 Generating training data...");
    let (train_data, val_data) = train::generate_training_data(all_words, num_games, samples_per_state);

    // Train
    println!("\n🏋️ Training model...");

    if use_gpu {
        train_on_gpu(&train_data, &val_data, epochs, batch_size, lr);
    } else {
        train_on_cpu(&train_data, &val_data, epochs, batch_size, lr);
    }
}

#[cfg(feature = "wgpu")]
fn train_on_gpu(train_data: &[train::TrainingSample], val_data: &[train::TrainingSample], epochs: usize, batch_size: usize, lr: f64) {
    println!("   Backend: WGPU (GPU/Metal)");
    type GpuBackend = burn::backend::Autodiff<burn::backend::Wgpu<f32>>;
    let device = burn::backend::wgpu::WgpuDevice::default();
    let model = train::train_model::<GpuBackend>(&device, train_data, val_data, epochs, batch_size, lr);
    train::save_model(&model, "artifacts/wordlebrain_model");
    println!("\n✅ Done! Model saved to artifacts/wordlebrain_model");
}

#[cfg(not(feature = "wgpu"))]
fn train_on_gpu(train_data: &[train::TrainingSample], val_data: &[train::TrainingSample], _epochs: usize, _batch_size: usize, _lr: f64) {
    eprintln!("⚠️  GPU backend not available. Build with --features wgpu to enable GPU training.");
    eprintln!("   Falling back to CPU (NdArray) backend.");
    train_on_cpu(train_data, val_data, _epochs, _batch_size, _lr);
}

fn train_on_cpu(train_data: &[train::TrainingSample], val_data: &[train::TrainingSample], epochs: usize, batch_size: usize, lr: f64) {
    println!("   Backend: NdArray (CPU)");
    type CpuBackend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let model = train::train_model::<CpuBackend>(&device, train_data, val_data, epochs, batch_size, lr);
train::save_model(&model, "artifacts/wordlebrain_model");
    println!("\n✅ Done! Model saved to artifacts/wordlebrain_model");
}

// ── Bench ───────────────────────────────────────────────────────────────────

#[allow(clippy::needless_range_loop)]
fn cmd_bench(all_words: &[String], count: usize, use_random: bool, use_model: bool, model_path: &str) {
    let solver_label = if use_model {
        "Neural Network"
    } else if use_random {
        "Random"
    } else {
        "Entropy"
    };

    println!(
        "\n📊 Benchmarking {} solver over {} random games...\n",
        solver_label, count
    );

    let model_solver: Option<ModelSolver> = if use_model {
        Some(ModelSolver::from_file(model_path))
    } else {
        None
    };

    let solver: &dyn Solver = if use_random { &RandomSolver } else { &EntropySolver };
    let mut total_guesses = 0;
    let mut wins = 0;
    let mut guess_distribution = [0usize; 7]; // 1..=6 guesses, 0=fail

    let mut solutions: Vec<String> = all_words.to_vec();
    solutions.shuffle(&mut rand::thread_rng());

    for (i, solution) in solutions.iter().take(count).enumerate() {
        let result = if let Some(ref ms) = model_solver {
            crate::solver::play_game(ms as &dyn Solver, solution, all_words, 6)
        } else {
            crate::solver::play_game(solver, solution, all_words, 6)
        };
        match result {
            Some(guesses) => {
                wins += 1;
                total_guesses += guesses;
                guess_distribution[guesses.min(6)] += 1;
            }
            None => {
                guess_distribution[0] += 1;
            }
        }

        if (i + 1) % 10 == 0 {
            println!("  Played {}/{} games...", i + 1, count);
        }
    }

    println!("\n📈 Results:");
    println!("   Win rate: {:.1}% ({}/{})", 100.0 * wins as f64 / count as f64, wins, count);
    if wins > 0 {
        println!("   Avg guesses (when won): {:.2}", total_guesses as f64 / wins as f64);
    }
    println!("   Guess distribution (1..6, 0=fail):");
    for g in 1..=6 {
        let bar = "█".repeat(guess_distribution[g].min(60));
        println!("     {}: {:>4} {}", g, guess_distribution[g], bar);
    }
    println!("     ✗: {:>4}", guess_distribution[0]);
}

// ── Display Helpers ─────────────────────────────────────────────────────────

fn print_pattern(pattern: &[Feedback; 5], word: &str) {
    let (colored, emoji) = pattern_parts(pattern, word);
    println!("   {}    {}", colored, emoji);
}

fn pattern_string(pattern: &[Feedback; 5]) -> String {
    pattern
        .iter()
        .map(|f| match f {
            Feedback::Green => "🟩",
            Feedback::Yellow => "🟨",
            Feedback::Grey => "⬛",
        })
        .collect::<Vec<_>>()
        .join("")
}

fn pattern_parts(pattern: &[Feedback; 5], word: &str) -> (String, String) {
    let chars: Vec<char> = word.chars().collect();
    let colored: String = chars
        .iter()
        .enumerate()
        .map(|(i, &c)| match pattern[i] {
            Feedback::Green => format!("\x1b[32m{}\x1b[0m", c),
            Feedback::Yellow => format!("\x1b[33m{}\x1b[0m", c),
            Feedback::Grey => format!("\x1b[37m{}\x1b[0m", c),
        })
        .collect::<Vec<_>>()
        .join(" ");
    let emoji = pattern_string(pattern);
    (colored, emoji)
}
