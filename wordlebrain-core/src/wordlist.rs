use crate::feedback::{is_consistent, Pattern};

/// Load the word list embedded at compile time.
/// Each word is a 5-letter lowercase string, one per line.
pub fn load_words() -> Vec<String> {
    include_str!("../../src/words")
        .lines()
        .map(|l: &str| l.trim().to_string())
        .filter(|w: &String| w.len() == 5 && w.chars().all(|c: char| c.is_ascii_lowercase()))
        .collect()
}

/// Filter a list of candidate words, keeping only those consistent
/// with all previous (guess, pattern) feedback.
pub fn filter(words: &[String], history: &[(String, Pattern)]) -> Vec<String> {
    words
        .iter()
        .filter(|w| {
            history
                .iter()
                .all(|(guess, pattern)| is_consistent(w, guess, pattern))
        })
        .cloned()
        .collect()
}
