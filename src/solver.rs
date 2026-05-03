use std::collections::HashMap;

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::module::Module;
use crate::feedback::{evaluate, pattern_key, Pattern};
use crate::model;
use crate::wordlist;

/// Trait for a Wordle solving strategy.
pub trait Solver {
    /// Given remaining candidate words and full word list, pick the next guess.
    fn next_guess(
        &self,
        remaining: &[String],
        all_words: &[String],
        history: &[(String, Pattern)],
    ) -> String;
}

/// Solver that picks the word with maximum entropy (information gain).
pub struct EntropySolver;

impl Solver for EntropySolver {
    fn next_guess(
        &self,
        remaining: &[String],
        all_words: &[String],
        _history: &[(String, Pattern)],
    ) -> String {
        if remaining.is_empty() {
            return all_words.first().cloned().unwrap_or_default();
        }
        if remaining.len() <= 2 {
            return remaining[0].clone();
        }

        let mut best_word = remaining[0].clone();
        let mut best_entropy = f64::NEG_INFINITY;

        for guess in all_words {
            let e = compute_entropy(guess, remaining);
            if e > best_entropy {
                best_entropy = e;
                best_word = guess.clone();
            }
        }

        best_word
    }
}

/// Compute entropy for a guess against remaining solution candidates.
/// Entropy = -Σ (p_i * log2(p_i)) where p_i is proportion of solutions in pattern group i.
pub fn compute_entropy(guess: &str, remaining: &[String]) -> f64 {
    let total = remaining.len() as f64;
    let mut counts: HashMap<u8, u32> = HashMap::new();

    for solution in remaining {
        let pattern = evaluate(guess, solution);
        *counts.entry(pattern_key(&pattern)).or_insert(0) += 1;
    }

    let mut entropy = 0.0;
    for &count in counts.values() {
        let p = count as f64 / total;
        entropy -= p * p.log2();
    }
    entropy
}

/// Baseline solver that picks a random word from remaining candidates.
pub struct RandomSolver;

impl Solver for RandomSolver {
    fn next_guess(
        &self,
        remaining: &[String],
        _all_words: &[String],
        _history: &[(String, Pattern)],
    ) -> String {
        use rand::seq::SliceRandom;
        if remaining.is_empty() {
            return String::new();
        }
        remaining
            .choose(&mut rand::thread_rng())
            .cloned()
            .unwrap_or_default()
    }
}

/// Play a full game using the given solver, returning the number of guesses used.
/// Returns None if the solver failed to solve within max_guesses.
pub fn play_game(
    solver: &dyn Solver,
    solution: &str,
    all_words: &[String],
    max_guesses: usize,
) -> Option<usize> {
    let mut remaining = all_words.to_vec();
    let mut history: Vec<(String, Pattern)> = Vec::new();

    for turn in 0..max_guesses {
        let guess = solver.next_guess(&remaining, all_words, &history);
        let pattern = crate::feedback::evaluate(&guess, solution);
        history.push((guess.clone(), pattern));

        if pattern.iter().all(|f| matches!(f, crate::feedback::Feedback::Green)) {
            return Some(turn + 1);
        }

        remaining = wordlist::filter(&remaining, &history);
    }

    None
}

// ── Model-Based Solver ──────────────────────────────────────────────────────

/// Solver that uses a trained neural network model to score candidate words.
/// The model is loaded from disk and used for inference only (no training needed).
pub struct ModelSolver {
    model: model::WordleModel<NdArray<f32>>,
    device: NdArrayDevice,
}

impl ModelSolver {
    /// Create a new ModelSolver from an already-loaded model.
    pub fn new(model: model::WordleModel<NdArray<f32>>, device: NdArrayDevice) -> Self {
        Self { model, device }
    }

    /// Load a ModelSolver from a saved model file.
    pub fn from_file(path: &str) -> Self {
        use burn::record::{CompactRecorder, Recorder};
        use crate::model::WordleModelConfig;

        let device = NdArrayDevice::default();
        let config = WordleModelConfig::new();
        let record = CompactRecorder::new()
            .load(path.into(), &device)
            .expect("Failed to load model record");
        let model = config.init(&device).load_record(record);
        Self { model, device }
    }
}

impl Solver for ModelSolver {
    fn next_guess(
        &self,
        remaining: &[String],
        _all_words: &[String],
        history: &[(String, Pattern)],
    ) -> String {
        if remaining.is_empty() {
            return String::new();
        }
        if remaining.len() <= 2 {
            return remaining[0].clone();
        }

        let state_features = model::encode_state(history);

        let mut best_word = remaining[0].clone();
        let mut best_score = f32::NEG_INFINITY;

        for word in remaining {
            let wf = model::encode_word(word);
            let score = model::score_word(&self.model, &state_features, &wf, &self.device);
            if score > best_score {
                best_score = score;
                best_word = word.clone();
            }
        }

        best_word
    }
}
