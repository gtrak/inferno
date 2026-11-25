//! The model tokenizer: `tokenizer.json` from the Hugging Face `tokenizers`
//! crate, with the two operations the server needs.
//!
//! Encoding is `add_special_tokens = false` (this checkpoint's
//! `tokenizer_config.json` sets `add_bos_token: false`, so the chat template owns
//! every special token), and decoding skips special tokens so `<|im_end|>` and
//! friends never reach the client.

use std::path::{Path, PathBuf};

use tokenizers::Tokenizer as HfTokenizer;

/// Anything that goes wrong loading or running the tokenizer.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    /// `tokenizer.json` could not be read.
    #[error("cannot read tokenizer at {path}: {source}")]
    Read {
        /// The path that failed.
        path: String,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// `tokenizer.json` is not a valid tokenizer document.
    #[error("cannot load tokenizer at {path}: {message}")]
    Load {
        /// The path that failed.
        path: String,
        /// What the `tokenizers` crate reported.
        message: String,
    },
    /// The tokenizer was loaded but refused to encode.
    #[error("tokenizer failed to encode the prompt: {0}")]
    Encode(String),
}

/// The model's tokenizer.
pub struct Tokenizer {
    inner: HfTokenizer,
    eos_token_id: Option<u32>,
}

impl Tokenizer {
    /// Loads `tokenizer.json`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(TokenizerError::Read {
                path: path.display().to_string(),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "no such file"),
            });
        }
        let inner = HfTokenizer::from_file(path).map_err(|error| TokenizerError::Load {
            path: path.display().to_string(),
            message: error.to_string(),
        })?;
        Ok(Self {
            inner,
            eos_token_id: None,
        })
    }

    /// Loads `tokenizer.json` from a model directory and resolves the end-of-turn
    /// token from the sibling `tokenizer_config.json`.
    pub fn from_model_directory(directory: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let directory = directory.as_ref();
        let mut tokenizer = Self::from_file(directory.join("tokenizer.json"))?;
        tokenizer.eos_token_id = read_eos_token(directory.join("tokenizer_config.json"))
            .and_then(|token| tokenizer.token_id(&token));
        Ok(tokenizer)
    }

    /// Wraps an already loaded tokenizer.
    pub fn from_inner(inner: HfTokenizer) -> Self {
        Self {
            inner,
            eos_token_id: None,
        }
    }

    /// The model's end-of-turn token, when the tokenizer config named one.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// Encodes `text` into vocabulary ids, without adding special tokens.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|error| TokenizerError::Encode(error.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decodes ids back into text, dropping special tokens.
    pub fn decode(&self, ids: &[u32]) -> String {
        match self.inner.decode(ids, true) {
            Ok(text) => text,
            Err(error) => {
                tracing::error!(%error, "tokenizer failed to decode generated ids");
                String::new()
            }
        }
    }

    /// The id `text` encodes to when it is exactly one token.
    pub fn token_id(&self, text: &str) -> Option<u32> {
        let ids = self.encode(text).ok()?;
        match ids.as_slice() {
            [id] => Some(*id),
            _ => None,
        }
    }

    /// Size of the vocabulary, special tokens included.
    pub fn vocabulary_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

/// Reads `eos_token` from a `tokenizer_config.json`, if it is a plain string.
fn read_eos_token(path: PathBuf) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let config: serde_json::Value = serde_json::from_str(&text).ok()?;
    config
        .get("eos_token")
        .and_then(|token| token.as_str())
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed byte-level fixture tokenizer: one token per byte.
    fn fixture() -> Tokenizer {
        Tokenizer::from_file("tests/fixtures/model/tokenizer.json")
            .expect("the fixture tokenizer is committed")
    }

    #[test]
    fn encode_decode_round_trips_utf8() {
        let tokenizer = fixture();
        for text in ["Hello world", "line\nbreak", "héllo → wörld", ""] {
            let ids = tokenizer.encode(text).expect("encodes");
            assert_eq!(tokenizer.decode(&ids), text, "round trip of {text:?}");
        }
    }

    #[test]
    fn encode_is_one_token_per_byte() {
        let tokenizer = fixture();
        let ids = tokenizer.encode("Hi").expect("encodes");
        assert_eq!(ids.len(), 2);
        assert_eq!(tokenizer.decode(&ids), "Hi");
    }

    #[test]
    fn special_tokens_are_skipped_when_decoding() {
        let tokenizer = fixture();
        let im_end = tokenizer
            .token_id("<|im_end|>")
            .expect("known special token");
        let text_ids = tokenizer.encode("ok").expect("encodes");
        let mut ids = text_ids.clone();
        ids.push(im_end);
        assert_eq!(tokenizer.decode(&ids), "ok");
    }
}
