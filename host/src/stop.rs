//! Stop-string handling on the *decoded text*, not per token.
//!
//! A stop string can span several tokens, and a token can end in the middle of a
//! multi-byte character, so the matcher works on the text the tokenizer produced:
//! it holds back the longest suffix that could still grow into a stop string,
//! emits everything before it, and never emits the stop text itself.

/// Matches stop strings in an incremental decoded stream.
pub struct StopMatcher {
    stops: Vec<String>,
    pending: String,
}

impl StopMatcher {
    /// Builds a matcher; empty stop strings are dropped because they would match
    /// everything immediately.
    pub fn new(stops: Vec<String>) -> Self {
        let stops: Vec<String> = stops.into_iter().filter(|stop| !stop.is_empty()).collect();
        Self {
            stops,
            pending: String::new(),
        }
    }

    /// True when there is nothing to match, so callers can skip the bookkeeping.
    pub fn is_empty(&self) -> bool {
        self.stops.is_empty()
    }

    /// Feeds newly decoded text.
    ///
    /// Returns the text that is safe to send and whether a stop string completed.
    /// Once this returns `true` the caller must stop feeding and drop the rest.
    pub fn feed(&mut self, text: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (text.to_owned(), false);
        }
        self.pending.push_str(text);

        if let Some(index) = self.first_stop_offset() {
            let emit = self.pending[..index].to_owned();
            self.pending.clear();
            return (emit, true);
        }

        let hold = self.hold_back();
        let split = self.pending.len() - hold;
        let emit = self.pending[..split].to_owned();
        self.pending.drain(..split);
        (emit, false)
    }

    /// Releases the held-back suffix when the generation ended without a stop.
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    /// Byte offset of the earliest complete stop string, if any.
    fn first_stop_offset(&self) -> Option<usize> {
        self.stops
            .iter()
            .filter_map(|stop| self.pending.find(stop.as_str()))
            .min()
    }

    /// Bytes at the end of `pending` that are a proper prefix of a stop string.
    fn hold_back(&self) -> usize {
        let longest = self.stops.iter().map(String::len).max().unwrap_or(0);
        let limit = longest.saturating_sub(1).min(self.pending.len());
        (1..=limit)
            .rev()
            .find(|length| {
                let start = self.pending.len() - length;
                self.pending.is_char_boundary(start)
                    && self
                        .stops
                        .iter()
                        .any(|stop| stop.starts_with(&self.pending[start..]))
            })
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_text_that_cannot_be_a_stop() {
        let mut matcher = StopMatcher::new(vec!["END".to_owned()]);
        assert_eq!(matcher.feed("hello "), ("hello ".to_owned(), false));
        assert_eq!(matcher.feed("world"), ("world".to_owned(), false));
    }

    #[test]
    fn a_stop_string_spanning_feeds_never_leaks() {
        let mut matcher = StopMatcher::new(vec!["\n\nUser:".to_owned()]);
        let mut emitted = String::new();
        let mut stopped = false;
        for piece in ["\n", "\n", "Us", "er:", "and more"] {
            let (text, hit) = matcher.feed(piece);
            emitted.push_str(&text);
            if hit {
                stopped = true;
                break;
            }
        }
        assert!(stopped, "the stop string completed");
        assert_eq!(emitted, "");
    }

    #[test]
    fn a_partial_stop_prefix_is_held_back_then_released() {
        let mut matcher = StopMatcher::new(vec!["STOP".to_owned()]);
        let (text, stopped) = matcher.feed("a ST");
        assert_eq!(text, "a ");
        assert!(!stopped);
        let (text, stopped) = matcher.feed("OP");
        assert_eq!(text, "");
        assert!(stopped);
    }

    #[test]
    fn flush_returns_the_held_back_suffix() {
        let mut matcher = StopMatcher::new(vec!["STOP".to_owned()]);
        let (text, stopped) = matcher.feed("tail S");
        assert_eq!(text, "tail ");
        assert!(!stopped);
        assert_eq!(matcher.flush(), "S");
    }

    #[test]
    fn the_earliest_stop_wins() {
        let mut matcher = StopMatcher::new(vec!["bb".to_owned(), "aa".to_owned()]);
        let (text, stopped) = matcher.feed("xaaybb");
        assert_eq!(text, "x");
        assert!(stopped);
    }

    #[test]
    fn without_stops_text_passes_through() {
        let mut matcher = StopMatcher::new(vec![String::new()]);
        assert!(matcher.is_empty());
        assert_eq!(matcher.feed("anything"), ("anything".to_owned(), false));
    }
}
