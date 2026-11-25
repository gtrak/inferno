//! Chat-template goldens: our renderer against Python `jinja2`, and our tokenizer
//! against the real model tokenizer.
//!
//! `../host/tests/fixtures/chat_goldens.json` was produced by
//! `../host/tests/fixtures/make_fixtures.py`: `rendered` is the output of Python `jinja2`
//! (the reference implementation) on the model's own `chat_template.jinja`, and
//! `ids` is what the model's real `tokenizer.json` produces from that text. The
//! committed template copy is byte-identical to the model's, so the rendering
//! half of this test needs no model files.
//!
//! The exact-id half needs the real tokenizer, and the *committed* fixture
//! tokenizer (`../host/tests/fixtures/model/tokenizer.json`) cannot replace it: it is a
//! 256-symbol byte-level stand-in, while the golden ids reach 248069, so a
//! fallback to it would compare nothing. The policy is therefore
//!
//! * `INFERNO_MODEL_DIR` set, its `tokenizer.json` present → run, after checking
//!   the file against the hash the goldens record;
//! * `INFERNO_MODEL_DIR` set but its `tokenizer.json` missing → **fail**, naming
//!   the variable and the expected file: a path you configured is a promise;
//! * nothing configured and the default location empty → **skip loudly**, printing
//!   why and what to set, so a run that did not check the ids cannot be read as
//!   one that did.

use std::path::PathBuf;

use inferno_server::template::{ChatTemplate, RenderContext};
use inferno_server::tokenizer::Tokenizer;
use serde_json::Value;

/// The goldens, committed next to the fixture model directory the template
/// and tokenizer modules own (`inferno-host`).
const GOLDENS: &str = include_str!("../../host/tests/fixtures/chat_goldens.json");

/// The environment variable a model directory is configured with.
const MODEL_DIRECTORY_VARIABLE: &str = "INFERNO_MODEL_DIR";

/// The model's chat template, byte-identical to `chat_template.jinja`.
fn fixture_template() -> PathBuf {
    fixture_directory().join("chat_template.jinja")
}

/// The committed model-shaped fixture directory.
fn fixture_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../host/tests/fixtures/model")
}

/// What the exact-id goldens should do, given the environment.
#[derive(Debug, PartialEq, Eq)]
enum GoldenTokenizer {
    /// This tokenizer file is the one the goldens were produced with.
    Use(PathBuf),
    /// `INFERNO_MODEL_DIR` is set and its `tokenizer.json` is not there.
    ConfiguredButMissing(PathBuf),
    /// Nothing was configured and the default location has no tokenizer.
    NotConfigured(PathBuf),
}

/// Resolves the golden tokenizer without reading the process environment, so all
/// three outcomes are testable. `home` is the value of `HOME`, empty when unset.
fn golden_tokenizer(
    model_directory: Option<&str>,
    home: &str,
    has_file: impl Fn(&std::path::Path) -> bool,
) -> GoldenTokenizer {
    let directory = match model_directory {
        Some(configured) => PathBuf::from(configured),
        None => PathBuf::from(home).join("models/qwen38-flash-next-nvfp4"),
    };
    let path = directory.join("tokenizer.json");
    if has_file(&path) {
        GoldenTokenizer::Use(path)
    } else if model_directory.is_some() {
        GoldenTokenizer::ConfiguredButMissing(path)
    } else {
        GoldenTokenizer::NotConfigured(path)
    }
}

/// The refusal when a configured model directory has no tokenizer.
fn configured_but_missing_message(path: &std::path::Path) -> String {
    format!(
        "{MODEL_DIRECTORY_VARIABLE} is set but its tokenizer is missing: {} \n\
         unset {MODEL_DIRECTORY_VARIABLE} to skip the exact-id goldens, or point it at \
         the directory holding the model's tokenizer.json",
        path.display(),
    )
}

/// The loud skip message when nothing is configured.
fn not_configured_message(path: &std::path::Path) -> String {
    format!(
        "SKIP the exact-id goldens: no model tokenizer at {} \n\
         set {MODEL_DIRECTORY_VARIABLE} to the model directory to run them; the committed \n\
         fixture tokenizer is a 256-symbol byte-level stand-in and cannot produce these ids",
        path.display(),
    )
}

/// The golden cases.
fn cases() -> Vec<Value> {
    let document: Value = serde_json::from_str(GOLDENS).expect("the goldens are valid JSON");
    assert_eq!(
        document["provenance"]["encode"], "tokenizer.encode(text, add_special_tokens=False)",
        "the goldens must be produced with the same encode call this test uses"
    );
    let cases = document["cases"].as_array().expect("cases is an array");
    assert!(!cases.is_empty(), "there must be golden cases");
    cases.clone()
}

/// Renders one case with the committed template.
fn render(case: &Value) -> String {
    let template = ChatTemplate::from_file(fixture_template()).expect("the template compiles");
    let messages = case["messages"].clone();
    let tools = case.get("tools").cloned();
    template
        .render(&RenderContext {
            messages: &messages,
            tools: tools.as_ref(),
            add_generation_prompt: case["add_generation_prompt"]
                .as_bool()
                .expect("add_generation_prompt is a bool"),
            enable_thinking: case.get("enable_thinking").and_then(Value::as_bool),
        })
        .unwrap_or_else(|error| panic!("case {} failed to render: {error}", case["name"]))
}

#[test]
fn every_case_renders_byte_identically_to_python_jinja2() {
    for case in cases() {
        let expected = case["rendered"].as_str().expect("rendered is a string");
        let rendered = render(&case);
        assert_eq!(
            rendered, expected,
            "case {} does not match the jinja2 reference",
            case["name"]
        );
    }
}

#[test]
fn the_real_tokenizer_produces_the_golden_ids() {
    let source = golden_tokenizer(
        std::env::var(MODEL_DIRECTORY_VARIABLE).ok().as_deref(),
        std::env::var("HOME").unwrap_or_default().as_str(),
        |path| path.is_file(),
    );
    let path = match source {
        GoldenTokenizer::ConfiguredButMissing(missing) => {
            panic!("{}", configured_but_missing_message(&missing));
        }
        GoldenTokenizer::NotConfigured(missing) => {
            println!("{}", not_configured_message(&missing));
            return;
        }
        GoldenTokenizer::Use(found) => found,
    };

    // The goldens record the hash of the tokenizer they were produced from, so a
    // configured directory that holds a different file is a failure, not a pass.
    let document: Value = serde_json::from_str(GOLDENS).expect("the goldens are valid JSON");
    let expected_hash = document["provenance"]["tokenizer_sha256"]
        .as_str()
        .expect("a tokenizer hash");
    let bytes = std::fs::read(&path).expect("the model tokenizer is readable");
    assert_eq!(
        sha256_hex(&bytes),
        expected_hash,
        "{} is not the tokenizer the goldens were produced from",
        path.display(),
    );

    let tokenizer = Tokenizer::from_file(&path).expect("the model tokenizer loads");

    for case in cases() {
        let rendered = render(&case);
        let expected: Vec<u32> = case["ids"]
            .as_array()
            .expect("ids is an array")
            .iter()
            .map(|id| u32::try_from(id.as_u64().expect("ids are numbers")).expect("ids fit u32"))
            .collect();
        let encoded = tokenizer.encode(&rendered).expect("the prompt encodes");
        assert_eq!(
            encoded.len(),
            expected.len(),
            "case {} has a different token count",
            case["name"]
        );
        assert_eq!(encoded, expected, "case {} has different ids", case["name"]);
    }
}

#[test]
fn a_configured_model_directory_without_a_tokenizer_is_a_failure() {
    let source = golden_tokenizer(Some("/no/such/model"), "/home/tester", |_| false);
    let GoldenTokenizer::ConfiguredButMissing(path) = source else {
        panic!("a configured directory must not skip: {source:?}");
    };
    let message = configured_but_missing_message(&path);
    assert!(
        message.contains(MODEL_DIRECTORY_VARIABLE),
        "the refusal must name the variable: {message}"
    );
    assert!(
        message.contains("tokenizer.json"),
        "the refusal must name the missing file: {message}"
    );
    assert!(path.ends_with("no/such/model/tokenizer.json"));
}

#[test]
fn an_unconfigured_machine_skips_the_id_goldens_with_a_reason() {
    let source = golden_tokenizer(None, "/home/tester", |_| false);
    let GoldenTokenizer::NotConfigured(path) = source else {
        panic!("nothing was configured, so nothing may fail: {source:?}");
    };
    let message = not_configured_message(&path);
    assert!(
        message.starts_with("SKIP"),
        "a skip must be unmistakable: {message}"
    );
    assert!(
        message.contains(MODEL_DIRECTORY_VARIABLE),
        "the skip must say how to unskip: {message}"
    );
}

#[test]
fn a_tokenizer_that_is_present_is_used_whatever_its_source() {
    let directory = fixture_directory();
    let source = golden_tokenizer(directory.to_str(), "/home/tester", |path| path.is_file());
    assert!(
        matches!(source, GoldenTokenizer::Use(_)),
        "the fixture directory does hold a tokenizer.json: {source:?}"
    );
}

#[test]
fn the_fixture_tokenizer_cannot_be_the_golden_source() {
    // Why the id goldens fall back to a skip instead of to the committed
    // stand-in: the fixture vocabulary is 256 byte symbols, the golden ids are not.
    let document: Value = serde_json::from_str(GOLDENS).expect("the goldens are valid JSON");
    let fixture = std::fs::read(fixture_directory().join("tokenizer.json"))
        .expect("the fixture tokenizer is readable");
    let vocabulary: Value = serde_json::from_slice(&fixture).expect("the fixture is valid JSON");
    let symbols = vocabulary["model"]["vocab"]
        .as_object()
        .expect("the fixture has a vocab")
        .len();
    let largest = document["cases"]
        .as_array()
        .expect("cases is an array")
        .iter()
        .flat_map(|case| case["ids"].as_array().expect("ids is an array").clone())
        .map(|id| id.as_u64().expect("ids are numbers"))
        .max()
        .expect("the goldens hold ids");
    assert!(
        largest >= symbols as u64,
        "golden ids {largest} fit the fixture vocab of {symbols}; a fallback would be vacuous"
    );
}

#[test]
fn the_fixture_template_is_the_model_template() {
    // A silent drift between the fixture and the model's template would make the
    // goldens meaningless; the goldens record the hash they were produced from.
    let document: Value = serde_json::from_str(GOLDENS).expect("the goldens are valid JSON");
    let expected_hash = document["provenance"]["template_sha256"]
        .as_str()
        .expect("a template hash");
    let template = std::fs::read(fixture_template()).expect("the fixture template is readable");
    assert_eq!(
        sha256_hex(&template),
        expected_hash,
        "the committed template is not the one the goldens were produced from"
    );
}

/// SHA-256 of `bytes`, hex encoded.
///
/// The repository pins its dependency set, and a hash of a few kilobytes does not
/// justify a new crate, so the compression function is written out here.
fn sha256_hex(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_length = (bytes.len() as u64) * 8;
    let mut message = bytes.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_length.to_be_bytes());

    for block in message.chunks(64) {
        let mut schedule = [0_u32; 64];
        for (index, word) in block.chunks(4).enumerate() {
            schedule[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for index in 16..64 {
            let first = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let second = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(first)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(second);
        }

        let mut working = state;
        for index in 0..64 {
            let sigma_first = working[4].rotate_right(6)
                ^ working[4].rotate_right(11)
                ^ working[4].rotate_right(25);
            let choose = (working[4] & working[5]) ^ (!working[4] & working[6]);
            let temp_first = working[7]
                .wrapping_add(sigma_first)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(schedule[index]);
            let sigma_second = working[0].rotate_right(2)
                ^ working[0].rotate_right(13)
                ^ working[0].rotate_right(22);
            let majority =
                (working[0] & working[1]) ^ (working[0] & working[2]) ^ (working[1] & working[2]);
            let temp_second = sigma_second.wrapping_add(majority);

            working[7] = working[6];
            working[6] = working[5];
            working[5] = working[4];
            working[4] = working[3].wrapping_add(temp_first);
            working[3] = working[2];
            working[2] = working[1];
            working[1] = working[0];
            working[0] = temp_first.wrapping_add(temp_second);
        }
        for (slot, value) in state.iter_mut().zip(working.iter()) {
            *slot = slot.wrapping_add(*value);
        }
    }

    let digest: String = state.iter().map(|word| format!("{word:08x}")).collect();
    assert_eq!(digest.len(), 64, "a sha256 digest is 32 bytes");
    digest
}

#[test]
fn the_sha256_helper_matches_the_known_vectors() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}
