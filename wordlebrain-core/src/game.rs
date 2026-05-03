use crate::feedback::{evaluate, Pattern};

/// Represents a single Wordle game.
pub struct Game {
    pub solution: String,
    guesses: Vec<(String, Pattern)>,
    max_guesses: usize,
}

impl Game {
    /// Create a new game with the given solution word.
    pub fn new(solution: String) -> Self {
        Self {
            solution,
            guesses: Vec::new(),
            max_guesses: 6,
        }
    }

    /// Make a guess. Returns the feedback Pattern.
    pub fn guess(&mut self, word: &str) -> Pattern {
        let pattern = evaluate(word, &self.solution);
        self.guesses.push((word.to_string(), pattern));
        pattern
    }

    /// Return the history of (guess, pattern) pairs.
    pub fn history(&self) -> &[(String, Pattern)] {
        &self.guesses
    }

    /// True if the last guess was fully correct (all Green).
    pub fn is_won(&self) -> bool {
        self.guesses
            .last()
            .is_some_and(|(_, p)| p.iter().all(|f| matches!(f, crate::feedback::Feedback::Green)))
    }

    /// True if game is over (won or max guesses reached).
    pub fn is_finished(&self) -> bool {
        self.is_won() || self.guesses.len() >= self.max_guesses
    }

    /// Number of guesses made so far.
    pub fn turns_used(&self) -> usize {
        self.guesses.len()
    }
}
