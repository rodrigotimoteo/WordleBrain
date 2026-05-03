use wasm_bindgen::prelude::*;
use wordlebrain_core::feedback::Feedback;
use wordlebrain_core::solver::{compute_first_turn_cache, EntropySolver, Solver};
use wordlebrain_core::wordlist;

use std::collections::HashMap;
use std::cell::RefCell;

thread_local! {
    static WORD_LIST: RefCell<Option<Vec<String>>> = const { RefCell::new(None) };
    static FIRST_TURN_CACHE: RefCell<Option<HashMap<String, f64>>> = const { RefCell::new(None) };
}

fn get_words() -> Vec<String> {
    WORD_LIST.with(|w| {
        w.borrow().clone().expect("call init() first")
    })
}

fn get_or_compute_cache() -> HashMap<String, f64> {
    FIRST_TURN_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        if let Some(ref m) = *cache {
            return m.clone();
        }
        let words = get_words();
        let computed = compute_first_turn_cache(&words);
        *cache = Some(computed.clone());
        computed
    })
}

/// Initialize the word list. Call once before using other functions.
/// Returns word count. The entropy cache is computed lazily on first use.
#[wasm_bindgen]
pub fn init() -> usize {
    let words = wordlist::load_words();
    let count = words.len();
    WORD_LIST.with(|w| *w.borrow_mut() = Some(words));
    count
}

/// Evaluate a guess against a solution. Returns a 5-character string:
/// G = Green, Y = Yellow, _ = Grey
#[wasm_bindgen]
pub fn evaluate(guess: &str, solution: &str) -> String {
    let pattern = wordlebrain_core::feedback::evaluate(guess, solution);
    pattern
        .iter()
        .map(|f| match f {
            Feedback::Green => 'G',
            Feedback::Yellow => 'Y',
            Feedback::Grey => '_',
        })
        .collect()
}

/// Get a random solution word from the word list.
#[wasm_bindgen]
pub fn random_word() -> String {
    use rand::seq::SliceRandom;
    let words = get_words();
    words
        .choose(&mut rand::thread_rng())
        .cloned()
        .unwrap_or_default()
}

/// Check if a word is in the word list.
#[wasm_bindgen]
pub fn validate_word(word: &str) -> bool {
    get_words().contains(&word.to_lowercase())
}

/// Get the best next guess given the current game history.
/// history_json is a JSON array of {"guess": "crane", "pattern": "G_Y__"} objects.
/// Returns the best guess word.
#[wasm_bindgen]
pub fn get_hint(history_json: &str) -> String {
    let words = get_words();
    let cache = get_or_compute_cache();
    let solver = EntropySolver {
        first_turn_cache: cache,
    };

    let history: Vec<(String, Pattern)> = match parse_history(history_json) {
        Ok(h) => h,
        Err(_) => {
            return solver
                .next_guess(&words, &words, &[])
        }
    };

    let remaining = wordlebrain_core::wordlist::filter(&words, &history);
    if remaining.is_empty() {
        return words.first().cloned().unwrap_or_default();
    }
    solver.next_guess(&remaining, &words, &history)
}

/// Solve a word step by step. Given a solution word and a step index (0-based),
/// returns JSON: {"guess": "crane", "pattern": "G_Y__", "remaining": 234}
/// Returns empty string if step is out of range (game won or exhausted).
#[wasm_bindgen]
pub fn solve_step(solution: &str, step: usize) -> String {
    let words = get_words();
    let cache = get_or_compute_cache();
    let solver = EntropySolver {
        first_turn_cache: cache,
    };

    let mut remaining = words.to_vec();
    let mut history: Vec<(String, Pattern)> = Vec::new();

    for i in 0..=step {
        if remaining.is_empty() {
            return String::new();
        }
        let guess = solver.next_guess(&remaining, &words, &history);
        let pattern = wordlebrain_core::feedback::evaluate(&guess, solution);
        let won = pattern.iter().all(|f| matches!(f, Feedback::Green));

        history.push((guess.clone(), pattern));

        if i == step {
            let result = serde_json::json!({
                "guess": guess,
                "pattern": pattern.iter().map(|f| match f {
                    Feedback::Green => 'G',
                    Feedback::Yellow => 'Y',
                    Feedback::Grey => '_',
                }).collect::<String>(),
                "remaining": remaining.len(),
                "won": won,
            });
            return result.to_string();
        }

        if won {
            return String::new();
        }
        remaining = wordlebrain_core::wordlist::filter(&remaining, &history);
    }

    String::new()
}

/// Solve a word completely. Returns JSON array of steps:
/// [{"guess":"crane","pattern":"G_Y__"}, ...]
#[wasm_bindgen]
pub fn solve_full(solution: &str) -> String {
    let words = get_words();
    let cache = get_or_compute_cache();
    let solver = EntropySolver {
        first_turn_cache: cache,
    };

    let result = play_game_trace(&solver, solution, &words);
    serde_json::to_string(&result).unwrap_or_else(|_| "[]".to_string())
}

/// Get the number of words in the dictionary.
#[wasm_bindgen]
pub fn word_count() -> usize {
    get_words().len()
}

// ── Helpers ───────────────────────────────────────────────────────────────────

type Pattern = [Feedback; 5];

fn parse_history(json: &str) -> Result<Vec<(String, Pattern)>, String> {
    let items: Vec<serde_json::Value> = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let mut history = Vec::new();
    for item in &items {
        let guess = item["guess"].as_str().ok_or("missing guess")?.to_string();
        let pattern_str = item["pattern"].as_str().ok_or("missing pattern")?;
        let pattern = parse_pattern(pattern_str)?;
        history.push((guess, pattern));
    }
    Ok(history)
}

fn parse_pattern(s: &str) -> Result<Pattern, String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() != 5 {
        return Err(format!("pattern must be 5 chars, got {}", chars.len()));
    }
    let mut pattern = [Feedback::Grey; 5];
    for i in 0..5 {
        pattern[i] = match chars[i] {
            'G' => Feedback::Green,
            'Y' => Feedback::Yellow,
            '_' | '.' | 'X' | 'x' => Feedback::Grey,
            c => return Err(format!("invalid pattern char: {}", c)),
        };
    }
    Ok(pattern)
}

#[derive(serde::Serialize)]
struct Step {
    guess: String,
    pattern: String,
}

fn play_game_trace(solver: &dyn Solver, solution: &str, all_words: &[String]) -> Vec<Step> {
    let mut remaining = all_words.to_vec();
    let mut history: Vec<(String, Pattern)> = Vec::new();
    let mut steps = Vec::new();

    for _ in 0..6 {
        if remaining.is_empty() {
            break;
        }
        let guess = solver.next_guess(&remaining, all_words, &history);
        let pattern = wordlebrain_core::feedback::evaluate(&guess, solution);
        let pattern_str = pattern
            .iter()
            .map(|f| match f {
                Feedback::Green => 'G',
                Feedback::Yellow => 'Y',
                Feedback::Grey => '_',
            })
            .collect::<String>();

        steps.push(Step {
            guess: guess.clone(),
            pattern: pattern_str,
        });

        history.push((guess, pattern));

        if pattern.iter().all(|f| matches!(f, Feedback::Green)) {
            break;
        }
        remaining = wordlebrain_core::wordlist::filter(&remaining, &history);
    }

    steps
}