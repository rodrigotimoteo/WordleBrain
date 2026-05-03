use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::module::Module;
use wordlebrain_core::feedback::Pattern;
use wordlebrain_core::solver::compute_entropy_raw;

use crate::model;

// ── First-Turn Entropy Cache (CLI: disk-persisted) ───────────────────────────

const CACHE_PATH: &str = "artifacts/first_turn_cache.json";

static FIRST_TURN_CACHE: OnceLock<Mutex<Option<HashMap<String, f64>>>> = OnceLock::new();

fn cache_lock() -> &'static Mutex<Option<HashMap<String, f64>>> {
    FIRST_TURN_CACHE.get_or_init(|| Mutex::new(None))
}

fn load_cache_from_disk() -> Option<HashMap<String, f64>> {
    let data = std::fs::read_to_string(CACHE_PATH).ok()?;
    serde_json::from_str(&data).ok()
}

fn save_cache_to_disk(map: &HashMap<String, f64>) {
    if let Ok(json) = serde_json::to_string(map) {
        let _ = std::fs::create_dir_all("artifacts");
        let _ = std::fs::write(CACHE_PATH, json);
        eprintln!("  💾 Cache saved to {}", CACHE_PATH);
    }
}

fn compute_first_turn_cache_cli(all_words: &[String]) -> HashMap<String, f64> {
    eprintln!(
        "  🔥 Computing first-turn entropy cache for {} words...",
        all_words.len()
    );
    let now = std::time::Instant::now();
    let map = wordlebrain_core::solver::compute_first_turn_cache(all_words);
    eprintln!("  ✅ Cache ready in {:.1}s", now.elapsed().as_secs_f64());
    map
}

/// Get the first-turn entropy for a word (CLI: uses disk cache).
pub fn first_turn_entropy(word: &str, all_words: &[String]) -> f64 {
    let cache = cache_lock();
    let mut guard = cache.lock().expect("cache lock poisoned");

    if let Some(ref map) = *guard {
        return map.get(word).copied().unwrap_or(0.0);
    }

    if let Some(map) = load_cache_from_disk() {
        eprintln!("  📂 Loaded first-turn cache from disk ({} words)", map.len());
        let result = map.get(word).copied().unwrap_or(0.0);
        *guard = Some(map);
        return result;
    }

    let map = compute_first_turn_cache_cli(all_words);
    save_cache_to_disk(&map);
    let result = map[word];
    *guard = Some(map);
    result
}

// ── Re-exports from core ─────────────────────────────────────────────────────

pub use wordlebrain_core::solver::{RandomSolver, Solver};

/// CLI entropy solver that uses disk-persisted first-turn cache.
/// Delegates to core EntropySolver but wraps the cache logic.
pub struct EntropySolver;

impl Solver for EntropySolver {
    fn next_guess(
        &self,
        remaining: &[String],
        all_words: &[String],
        history: &[(String, Pattern)],
    ) -> String {
        if remaining.is_empty() {
            return all_words.first().cloned().unwrap_or_default();
        }
        if remaining.len() <= 2 {
            return remaining[0].clone();
        }

        if history.is_empty() {
            let mut best_word = remaining[0].clone();
            let mut best_entropy = f64::NEG_INFINITY;
            for guess in all_words {
                let e = first_turn_entropy(guess, all_words);
                if e > best_entropy {
                    best_entropy = e;
                    best_word = guess.clone();
                }
            }
            return best_word;
        }

        let mut best_word = remaining[0].clone();
        let mut best_entropy = f64::NEG_INFINITY;

        for guess in all_words {
            let e = compute_entropy_raw(guess, remaining);
            if e > best_entropy {
                best_entropy = e;
                best_word = guess.clone();
            }
        }

        best_word
    }
}

/// Play a full game (re-exported from core).
pub use wordlebrain_core::solver::play_game;

// ── Model-Based Solver ───────────────────────────────────────────────────────

pub struct ModelSolver {
    model: model::WordleModel<NdArray<f32>>,
    device: NdArrayDevice,
}

impl ModelSolver {
    #[allow(dead_code)]
    pub fn new(model: model::WordleModel<NdArray<f32>>, device: NdArrayDevice) -> Self {
        Self { model, device }
    }

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