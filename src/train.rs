use burn::{
    module::Module,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use rand::seq::SliceRandom;

use crate::feedback::{evaluate, Feedback, Pattern};
use crate::model::{encode_state, encode_word, WordleModel, WordleModelConfig};
use crate::solver::{compute_entropy_raw, first_turn_entropy, EntropySolver, Solver};
use crate::wordlist;

// ── Training Sample ─────────────────────────────────────────────────────────

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrainingSample {
    pub state: Vec<f32>,
    pub word: Vec<f32>,
    pub entropy: f32,
}

// ── Data Generation ─────────────────────────────────────────────────────────

/// Generate training data by running the entropy solver on random games.
/// For each game turn, sample `samples_per_state` random words and compute their entropy.
pub fn generate_training_data(
    all_words: &[String],
    num_games: usize,
    samples_per_state: usize,
) -> Vec<TrainingSample> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::new();

    for game_idx in 0..num_games {
        let solution = all_words.choose(&mut rng).unwrap().clone();
        let mut remaining = all_words.to_vec();
        let mut history: Vec<(String, Pattern)> = Vec::new();
        let solver = EntropySolver;

        for _turn in 0..6 {
            let state_features = encode_state(&history);

            // Sample random candidate words and compute their entropy
            let candidates: Vec<&String> = if samples_per_state < all_words.len() {
                all_words.choose_multiple(&mut rng, samples_per_state).collect()
            } else {
                all_words.iter().collect()
            };

            for candidate in &candidates {
                let e = if history.is_empty() {
                    first_turn_entropy(candidate, all_words)
                } else {
                    compute_entropy_raw(candidate, &remaining)
                };
                let word_features = encode_word(candidate);
                data.push(TrainingSample {
                    state: state_features.clone(),
                    word: word_features,
                    entropy: e as f32,
                });
            }

            // Use entropy solver to advance the game
            let guess = solver.next_guess(&remaining, all_words, &history);
            let pattern = evaluate(&guess, &solution);
            history.push((guess.clone(), pattern));

            if pattern.iter().all(|f| matches!(f, Feedback::Green)) {
                break;
            }
            remaining = wordlist::filter(&remaining, &history);
        }

        if (game_idx + 1) % 10 == 0 {
            println!(
                "  Generated data from {} / {} games ({} samples so far)",
                game_idx + 1,
                num_games,
                data.len()
            );
        }
    }

    // Shuffle
    data.shuffle(&mut rng);
    println!("  Total training samples: {}", data.len());
    data
}

// ── Training ────────────────────────────────────────────────────────────────

/// Train a WordleModel using the provided training samples.
/// Returns the trained model.
pub fn train_model<B: AutodiffBackend<FloatElem = f32>>(
    device: &B::Device,
    data: &[TrainingSample],
    num_epochs: usize,
    batch_size: usize,
    learning_rate: f64,
) -> WordleModel<B> {
    let config = WordleModelConfig::new();
    let mut model = config.init(device);
    let mut optim = AdamConfig::new().init::<B, WordleModel<B>>();

    println!(
        "  Training: {} samples, {} epochs, batch_size={}, lr={}",
        data.len(),
        num_epochs,
        batch_size,
        learning_rate
    );

    for epoch in 0..num_epochs {
        let mut total_loss: f32 = 0.0;
        let mut batch_count = 0usize;

        for chunk in data.chunks(batch_size) {
            if chunk.is_empty() {
                continue;
            }

            // Build input batch [batch_size, INPUT_DIM]
            let mut input_tensors: Vec<Tensor<B, 2>> = Vec::with_capacity(chunk.len());
            for sample in chunk {
                let mut combined = sample.state.clone();
                combined.extend_from_slice(&sample.word);
                input_tensors.push(
                    Tensor::<B, 1>::from_floats(&combined[..], device).unsqueeze::<2>(),
                );
            }
            let inputs = Tensor::cat(input_tensors, 0);

            // Build target batch [batch_size]
            let mut target_tensors: Vec<Tensor<B, 1>> = Vec::with_capacity(chunk.len());
            for sample in chunk {
                target_tensors.push(Tensor::<B, 1>::from_floats([sample.entropy], device));
            }
            let targets = Tensor::cat(target_tensors, 0);

            // Forward
            let output = model.forward(inputs); // [batch, 1]

            // MSE loss
            let loss = (output.clone() - targets.clone().unsqueeze::<2>())
                .powf_scalar(2.0)
                .mean();

            // Backward + optimize
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optim.step(learning_rate, model, grads_params);

            total_loss += loss.into_scalar();
            batch_count += 1;
        }

        if (epoch + 1) % 10 == 0 || epoch == 0 {
            println!(
                "  Epoch {}/{} - avg loss: {:.6}",
                epoch + 1,
                num_epochs,
                total_loss / batch_count as f32
            );
        }
    }

    println!("  Training complete.");
    model
}

/// Save the trained model to a file using Burn's CompactRecorder.
pub fn save_model<B: Backend>(model: &WordleModel<B>, path: &str) {
    use burn::record::{CompactRecorder, Recorder};
    CompactRecorder::new()
        .record(model.clone().into_record(), path.into())
        .expect("Failed to save model");
    println!("  Model saved to {}", path);
}

/// Load a trained model from a file.
pub fn load_model<B: Backend>(device: &B::Device, path: &str) -> WordleModel<B> {
    use burn::record::{CompactRecorder, Recorder};
    let config = WordleModelConfig::new();
    let record = CompactRecorder::new()
        .load(path.into(), device)
        .expect("Failed to load model record");
    config.init(device).load_record(record)
}
