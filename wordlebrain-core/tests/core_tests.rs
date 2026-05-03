use wordlebrain_core::feedback::{evaluate, is_consistent, pattern_from_key, pattern_key, Feedback};
use wordlebrain_core::game::Game;
use wordlebrain_core::solver::{compute_entropy_raw, RandomSolver};
use wordlebrain_core::wordlist;

mod feedback {
    use super::*;

    #[test]
    fn all_green() {
        let p = evaluate("crane", "crane");
        assert!(p.iter().all(|f| matches!(f, Feedback::Green)));
    }

    #[test]
    fn no_shared_letters() {
        let p = evaluate("abcde", "fghij");
        assert!(p.iter().all(|f| matches!(f, Feedback::Grey)));
    }

#[test]
    fn simple_yellow() {
        let p = evaluate("crane", "stare");
        // c-vs-s:Grey(G), r-vs-t:Yellow(r in sol at pos3), a-vs-a:Green, n-vs-r:Grey, e-vs-e:Green
        assert_eq!(p[0], Feedback::Grey);
        assert_eq!(p[1], Feedback::Yellow);
        assert_eq!(p[2], Feedback::Green);
        assert_eq!(p[3], Feedback::Grey);
        assert_eq!(p[4], Feedback::Green);
    }

#[test]
    fn duplicate_letters_green_and_grey() {
        // speed vs steep (s,t,e,e,p)
        // Pass1 greens: s(0)=s(0), e(2)=e(2), e(3)=e(3) -> all Green
        // Pass2: p(1) -> p at sol pos4 unused -> Yellow
        //        d(4) -> not in unused sol -> Grey
        let p = evaluate("speed", "steep");
        assert_eq!(p[0], Feedback::Green);  // s=s
        assert_eq!(p[1], Feedback::Yellow);  // p matches p(4) in wrong pos
        assert_eq!(p[2], Feedback::Green);  // e=e(2)
        assert_eq!(p[3], Feedback::Green);  // e=e(3)
        assert_eq!(p[4], Feedback::Grey);  // d not in steep
    }

    #[test]
    fn duplicate_in_guess_single_in_solution() {
        let p = evaluate("llama", "crane");
        assert_eq!(p[0], Feedback::Grey);
        assert_eq!(p[1], Feedback::Grey);
        assert_eq!(p[2], Feedback::Green); // a at pos 2
        assert_eq!(p[3], Feedback::Grey);
        assert_eq!(p[4], Feedback::Grey);
    }

    #[test]
    fn pattern_key_roundtrip() {
        let p = [
            Feedback::Green,
            Feedback::Yellow,
            Feedback::Grey,
            Feedback::Green,
            Feedback::Grey,
        ];
        let key = pattern_key(&p);
        let decoded = pattern_from_key(key);
        assert_eq!(p, decoded);
    }

    #[test]
    fn pattern_key_all_green() {
        let p = [Feedback::Green; 5];
        assert_eq!(pattern_key(&p), 0);
    }

    #[test]
    fn pattern_key_all_grey() {
        let p = [Feedback::Grey; 5];
        assert_eq!(pattern_key(&p), 242); // 2*3^4 + 2*3^3 + 2*3^2 + 2*3^1 + 2*3^0
    }

    #[test]
    fn is_consistent_green() {
        let pattern = evaluate("crane", "crane");
        assert!(is_consistent("crane", "crane", &pattern));
        assert!(!is_consistent("stare", "crane", &pattern)); // s!=c at pos 0
    }

    #[test]
    fn is_consistent_yellow() {
        let pattern = evaluate("stare", "crane");
        // 'stare' guessed against 'crane': S=Grey, T=Grey, A=Green, R=Yellow, E=Green
        assert!(is_consistent("crane", "stare", &pattern));
    }

    #[test]
    fn is_consistent_grey_means_no_more() {
        let pattern = evaluate("bbbbb", "aaaaa");
        // all grey — word must not contain 'b'
        assert!(!is_consistent("babel", "bbbbb", &pattern));
    }
}

mod game {
    use super::*;

    #[test]
    fn new_game_not_won() {
        let g = Game::new("crane".to_string());
        assert!(!g.is_won());
        assert!(!g.is_finished());
    }

    #[test]
    fn winning_guess() {
        let mut g = Game::new("crane".to_string());
        g.guess("crane");
        assert!(g.is_won());
        assert!(g.is_finished());
        assert_eq!(g.turns_used(), 1);
    }

    #[test]
    fn wrong_guess_not_won() {
        let mut g = Game::new("crane".to_string());
        g.guess("stare");
        assert!(!g.is_won());
        assert!(!g.is_finished());
    }

    #[test]
    fn six_wrong_guesses_finished() {
        let mut g = Game::new("crane".to_string());
        for _ in 0..6 {
            g.guess("stare");
        }
        assert!(!g.is_won());
        assert!(g.is_finished());
    }

    #[test]
    fn history_tracks_guesses() {
        let mut g = Game::new("crane".to_string());
        g.guess("stare");
        g.guess("crane");
        assert_eq!(g.history().len(), 2);
        assert_eq!(g.history()[0].0, "stare");
        assert_eq!(g.history()[1].0, "crane");
    }
}

mod wordlist_filter {
    use super::*;

    #[test]
    fn load_words_not_empty() {
        let words = wordlist::load_words();
        assert!(!words.is_empty());
        assert!(words.len() > 1000);
    }

    #[test]
    fn all_words_five_letters() {
        let words = wordlist::load_words();
        for w in &words {
            assert_eq!(w.len(), 5, "word '{}' is not 5 letters", w);
            assert!(w.chars().all(|c| c.is_ascii_lowercase()), "word '{}' has non-lowercase", w);
        }
    }

    #[test]
    fn filter_after_green() {
        let words = wordlist::load_words();
        let pattern = evaluate("crane", "crane"); // all green
        let filtered = wordlist::filter(&words, &[("crane".to_string(), pattern)]);
        assert!(filtered.contains(&"crane".to_string()));
        assert!(!filtered.contains(&"stare".to_string()));
    }

    #[test]
    fn filter_after_grey() {
        let words = wordlist::load_words();
        let pattern = evaluate("xyzwf", "crane"); // all grey
        let filtered = wordlist::filter(&words, &[("xyzwf".to_string(), pattern)]);
        // Should exclude words containing x, y, z, w, f
        for w in &filtered {
            assert!(!w.contains('x'), "filtered word '{}' contains 'x'", w);
            assert!(!w.contains('y'), "filtered word '{}' contains 'y'", w);
            assert!(!w.contains('z'), "filtered word '{}' contains 'z'", w);
            assert!(!w.contains('w'), "filtered word '{}' contains 'w'", w);
            assert!(!w.contains('f'), "filtered word '{}' contains 'f'", w);
        }
    }
}

mod solver {
    use super::*;
    use wordlebrain_core::solver::Solver;

    #[test]
    fn entropy_positive() {
        let words = vec!["crane".to_string(), "stare".to_string(), "slate".to_string()];
        let e = compute_entropy_raw("crane", &words);
        assert!(e > 0.0);
    }

    #[test]
    fn entropy_deterministic() {
        let words = vec!["crane".to_string(), "stare".to_string(), "slate".to_string()];
        let e1 = compute_entropy_raw("crane", &words);
        let e2 = compute_entropy_raw("crane", &words);
        assert!((e1 - e2).abs() < f64::EPSILON);
    }

    #[test]
    fn entropy_single_word_is_zero() {
        let words = vec!["crane".to_string()];
        let e = compute_entropy_raw("crane", &words);
        assert!((e - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn random_solver_returns_valid_word() {
        let words = vec!["crane".to_string(), "stare".to_string()];
        let solver = RandomSolver;
        let guess = solver.next_guess(&words, &words, &[]);
        assert!(words.contains(&guess));
    }
}