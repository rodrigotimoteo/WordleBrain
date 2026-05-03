#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Feedback {
    Green,
    Yellow,
    Grey,
}

/// A 5-element array representing feedback for each position of a 5-letter guess.
pub type Pattern = [Feedback; 5];

/// Encode a Pattern as a base-3 integer (0–242), where Green=0, Yellow=1, Grey=2.
/// This is useful for HashMap keys and compact storage.
pub fn pattern_key(pattern: &Pattern) -> u8 {
    let mut key = 0u8;
    for &f in pattern {
        key = key * 3
            + match f {
                Feedback::Green => 0,
                Feedback::Yellow => 1,
                Feedback::Grey => 2,
            };
    }
    key
}

/// Decode a base-3 key back into a Pattern.
pub fn pattern_from_key(mut key: u8) -> Pattern {
    let mut pattern = [Feedback::Grey; 5];
    for i in (0..5).rev() {
        let rem = key % 3;
        pattern[i] = match rem {
            0 => Feedback::Green,
            1 => Feedback::Yellow,
            _ => Feedback::Grey,
        };
        key /= 3;
    }
    pattern
}

/// Evaluate a guess against a solution, returning a Pattern.
///
/// Correctly handles duplicate letters:
/// 1. First pass: mark exact matches (Green) and "consume" those solution positions.
/// 2. Second pass: mark Yellow only if the letter appears in an unclaimed solution position.
/// 3. Everything else is Grey.
pub fn evaluate(guess: &str, solution: &str) -> Pattern {
    let g: Vec<char> = guess.chars().collect();
    let s: Vec<char> = solution.chars().collect();
    assert_eq!(g.len(), 5, "guess must be 5 letters");
    assert_eq!(s.len(), 5, "solution must be 5 letters");

    let mut result = [Feedback::Grey; 5];
    let mut sol_used = [false; 5];

    // Pass 1: Greens
    for i in 0..5 {
        if g[i] == s[i] {
            result[i] = Feedback::Green;
            sol_used[i] = true;
        }
    }

    // Pass 2: Yellows
    for i in 0..5 {
        if result[i] == Feedback::Green {
            continue;
        }
        for j in 0..5 {
            if !sol_used[j] && g[i] == s[j] {
                result[i] = Feedback::Yellow;
                sol_used[j] = true;
                break;
            }
        }
    }

    result
}

/// Check whether a candidate word is consistent with a (guess, pattern) pair.
pub fn is_consistent(word: &str, guess: &str, pattern: &Pattern) -> bool {
    let w: Vec<char> = word.chars().collect();
    let g: Vec<char> = guess.chars().collect();

    // Count occurrences in candidate word
    let mut word_counts = [0u8; 26];
    for &c in &w {
        word_counts[letter_idx(c)] += 1;
    }

    // Track minimum required and whether letter has any grey feedback
    let mut min_required = [0u8; 26];
    let mut has_grey = [false; 26];
    let mut exact_positions: [Option<char>; 5] = [None; 5];

    for i in 0..5 {
        let idx = letter_idx(g[i]);
        match pattern[i] {
            Feedback::Green => {
                exact_positions[i] = Some(g[i]);
                min_required[idx] += 1;
            }
            Feedback::Yellow => {
                min_required[idx] += 1;
                // Yellow at position i means word[i] MUST NOT be g[i]
                if w[i] == g[i] {
                    return false;
                }
            }
            Feedback::Grey => {
                has_grey[idx] = true;
            }
        }
    }

    // Check exact position constraints (Greens)
    for i in 0..5 {
        if let Some(c) = exact_positions[i] {
            if w[i] != c {
                return false;
            }
        }
    }

    // Check letter count constraints
    for idx in 0..26 {
        // Word must have at least min_required copies
        if word_counts[idx] < min_required[idx] {
            return false;
        }
        // If letter has grey feedback, word must not have extra copies
        if has_grey[idx] && word_counts[idx] > min_required[idx] {
            return false;
        }
    }

    true
}

fn letter_idx(c: char) -> usize {
    (c as u8 - b'a') as usize
}
