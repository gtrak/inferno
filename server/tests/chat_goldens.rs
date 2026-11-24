//! Chat-template goldens: our renderer against Python `jinja2`, and our tokenizer
//! against the real model tokenizer.
//!
//! `tests/fixtures/chat_goldens.json` was produced by
//! `tests/fixtures/make_fixtures.py`: `rendered` is the output of Python `jinja2`
//! (the reference implementation) on the model's own `chat_template.jinja`, and
//! `ids` is what the model's real `tokenizer.json` produces from that text. The
//! committed template copy is byte-identical to the model's, so the rendering
//! half of this test needs no model files; the exact-id half needs the real
//! tokenizer and fails loudly (it does not skip) when it is absent, because a
//! skipped golden test is not a gate.

use std::path::PathBuf;

use inferno_server::template::{ChatTemplate, RenderContext};
use inferno_server::tokenizer::Tokenizer;
use serde_json::Value;

/// The goldens, committed.
const GOLDENS: &str = include_str!("fixtures/chat_goldens.json");

/// The model's chat template, byte-identical to `chat_template.jinja`.
fn fixture_template() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/model/chat_template.jinja")
}

/// The real tokenizer, from `INFERNO_MODEL_DIR` or the default model location.
fn real_tokenizer_path() -> PathBuf {
    let directory = std::env::var("INFERNO_MODEL_DIR").unwrap_or_else(|_| {
        format!(
            "{}/models/qwen38-flash-next-nvfp4",
            std::env::var("HOME").expect("HOME is set")
        )
    });
    PathBuf::from(directory).join("tokenizer.json")
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
    let path = real_tokenizer_path();
    assert!(
        path.is_file(),
        "the model tokenizer is required for the exact-id goldens: {} is missing \
         (set INFERNO_MODEL_DIR to the model directory)",
        path.display()
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
