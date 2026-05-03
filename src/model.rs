use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
};

use wordlebrain_core::feedback::{Feedback, Pattern};

// ── Dimensions ──────────────────────────────────────────────────────────────

/// State features: 5*26 green + 5*26 yellow_pos + 26 required_counts + 26 grey_eliminated
pub const STATE_DIM: usize = 312;

/// Word features: 5*26 one-hot
pub const WORD_DIM: usize = 130;

/// Combined input dimension
pub const INPUT_DIM: usize = STATE_DIM + WORD_DIM; // 442

// ── Model ───────────────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct WordleModel<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    output: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct WordleModelConfig {
    #[config(default = 256)]
    pub hidden1: usize,
    #[config(default = 128)]
    pub hidden2: usize,
    #[config(default = 64)]
    pub hidden3: usize,
}

impl WordleModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> WordleModel<B> {
        WordleModel {
            linear1: LinearConfig::new(INPUT_DIM, self.hidden1).with_bias(true).init(device),
            linear2: LinearConfig::new(self.hidden1, self.hidden2).with_bias(true).init(device),
            linear3: LinearConfig::new(self.hidden2, self.hidden3).with_bias(true).init(device),
            output: LinearConfig::new(self.hidden3, 1).with_bias(true).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> WordleModel<B> {
    /// Forward pass: input [batch, INPUT_DIM] -> output [batch, 1]
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);
        let x = self.activation.forward(x);
        let x = self.linear3.forward(x);
        let x = self.activation.forward(x);
        self.output.forward(x)
    }
}

// ── Feature Encoding ────────────────────────────────────────────────────────

fn letter_idx(c: char) -> usize {
    (c as u8 - b'a') as usize
}

/// Encode the current game state (from history) into a 312-element f32 vector.
///
/// Layout:
///   [0..129]   Green: 5 positions × 26 letters (1.0 = known green at pos)
///   [130..259] Yellow: 5 positions × 26 letters (1.0 = this letter was yellow at this pos)
///   [260..285] Required: 26 values (min required copies / 5.0)
///   [286..311] Grey: 26 values (1.0 = fully eliminated, no green/yellow exceptions)
pub fn encode_state(history: &[(String, Pattern)]) -> Vec<f32> {
    let mut features = vec![0.0f32; STATE_DIM];
    if history.is_empty() {
        return features;
    }

    let mut idx = 0;
    let guesses: Vec<&str> = history.iter().map(|(g, _)| g.as_str()).collect();
    let patterns: Vec<&Pattern> = history.iter().map(|(_, p)| p).collect();

    // Green known positions: 5 × 26
    for pos in 0..5 {
        let mut known_green_letter: Option<char> = None;
        for (g, p) in guesses.iter().zip(&patterns) {
            if matches!(p[pos], Feedback::Green) {
                known_green_letter = Some(g.chars().nth(pos).unwrap());
                break; // only one green can be set per position
            }
        }
        for letter in 0..26 {
            if let Some(c) = known_green_letter
                && letter_idx(c) == letter
            {
                features[idx] = 1.0;
            }
            idx += 1;
        }
    }

    // Yellow seen at positions: 5 × 26
    for pos in 0..5 {
        for letter in 0..26 {
            let yellow_here = guesses.iter().zip(&patterns).any(|(g, p)| {
                matches!(p[pos], Feedback::Yellow)
                    && g.chars().nth(pos).is_some_and(|c| letter_idx(c) == letter)
            });
            features[idx] = if yellow_here { 1.0 } else { 0.0 };
            idx += 1;
        }
    }

    // Required letter counts: 26 (normalized by 5)
    for letter in 0..26 {
        let mut count: f32 = 0.0;
        for (g, p) in guesses.iter().zip(&patterns) {
            for i in 0..5 {
                if matches!(p[i], Feedback::Green | Feedback::Yellow)
                    && g.chars().nth(i).is_some_and(|c| letter_idx(c) == letter)
                {
                    count += 1.0;
                }
            }
        }
        features[idx] = count / 5.0;
        idx += 1;
    }

    // Grey eliminated letters: 26
    for letter in 0..26 {
        let has_grey = guesses.iter().zip(&patterns).any(|(g, p)| {
            g.chars().enumerate().any(|(i, c)| {
                matches!(p[i], Feedback::Grey) && letter_idx(c) == letter
            })
        });
        let has_green_yellow = guesses.iter().zip(&patterns).any(|(g, p)| {
            g.chars().enumerate().any(|(i, c)| {
                matches!(p[i], Feedback::Green | Feedback::Yellow)
                    && letter_idx(c) == letter
            })
        });
        features[idx] = if has_grey && !has_green_yellow { 1.0 } else { 0.0 };
        idx += 1;
    }

    features
}

/// Encode a candidate word into a 130-element one-hot vector.
/// Layout: 5 positions × 26 letters (1.0 at word[pos][letter]).
pub fn encode_word(word: &str) -> Vec<f32> {
    let mut features = vec![0.0f32; WORD_DIM];
    for (i, c) in word.chars().enumerate().take(5) {
        if c.is_ascii_lowercase() {
            let letter = letter_idx(c);
            features[i * 26 + letter] = 1.0;
        }
    }
    features
}

/// Score a (state, word) pair using the model. Higher = better guess.
/// Returns the raw model output as a scalar.
pub fn score_word<B: Backend<FloatElem = f32>>(
    model: &WordleModel<B>,
    state_features: &[f32],
    word_features: &[f32],
    device: &B::Device,
) -> f32 {
    let mut combined = state_features.to_vec();
    combined.extend_from_slice(word_features);
    let input = Tensor::<B, 1>::from_floats(&combined[..], device).unsqueeze::<2>();
    let output = model.forward(input);
    output.into_scalar()
}

/// Score all candidate words given a game state. Returns (word, score) sorted descending.
#[allow(dead_code)]
pub fn score_candidates<B: Backend<FloatElem = f32>>(
    model: &WordleModel<B>,
    candidates: &[String],
    state_features: &[f32],
    device: &B::Device,
) -> Vec<(String, f32)> {
    let mut scored: Vec<(String, f32)> = candidates
        .iter()
        .map(|word| {
            let wf = encode_word(word);
            let score = score_word(model, state_features, &wf, device);
            (word.clone(), score)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}
