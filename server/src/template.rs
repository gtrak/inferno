//! Chat-template rendering and tool-call parsing.
//!
//! The template is the model's own `chat_template.jinja`, rendered with
//! `minijinja`. It is Jinja, but not Jinja2-the-Python-library: it uses
//! `content.startswith(...)` / `content.endswith(...)`, which minijinja does not
//! implement as string methods, so [`shim_template`] rewrites those two calls into
//! function calls before compiling. Nothing else is rewritten, and
//! `tests/fixtures/chat_goldens.json` pins the rendered text of every supported
//! message shape against the Python `jinja2` rendering of the *unmodified*
//! template, so the shim is proven equivalent rather than assumed.
//!
//! The `tojson` filter is reimplemented ([`python_tojson`]) because the template
//! puts tool schemas into the prompt and the prompt must match Hugging Face byte
//! for byte: Python's `jinja2` serialises with sorted keys, `", "` / `": "`
//! separators, `ensure_ascii` and HTML-safe escaping of `<`, `>`, `&` and `'`.

use std::path::Path;

use minijinja::value::Value;
use serde::Serialize as _;

/// Opening marker of a tool call in the model's output.
const TOOL_CALL_OPEN: &str = "<tool_call>";
/// Closing marker of a tool call in the model's output.
const TOOL_CALL_CLOSE: &str = "</tool_call>";
/// Opening marker of the function name inside a tool call.
const FUNCTION_OPEN: &str = "<function=";
/// Closing marker of the function body inside a tool call.
const FUNCTION_CLOSE: &str = "</function>";
/// Opening marker of one argument inside a tool call.
const PARAMETER_OPEN: &str = "<parameter=";
/// Closing marker of one argument inside a tool call.
const PARAMETER_CLOSE: &str = "</parameter>";

/// Anything that goes wrong loading or rendering the template.
#[derive(Debug, thiserror::Error)]
pub enum TemplateError {
    /// `chat_template.jinja` could not be read.
    #[error("cannot read chat template at {path}: {source}")]
    Read {
        /// The path that failed.
        path: String,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// The template does not compile.
    #[error("chat template is invalid: {0}")]
    Compile(String),
    /// Rendering failed, usually because the messages do not fit the template.
    #[error("{0}")]
    Render(String),
}

/// What the template is rendered with.
pub struct RenderContext<'a> {
    /// The `messages` array, in wire form.
    pub messages: &'a serde_json::Value,
    /// The `tools` array, in wire form, when the client sent one.
    pub tools: Option<&'a serde_json::Value>,
    /// Appends the assistant prefix the model continues from.
    pub add_generation_prompt: bool,
    /// `None` leaves the template's own default (thinking enabled).
    pub enable_thinking: Option<bool>,
}

/// The model's chat template, compiled once at startup.
pub struct ChatTemplate {
    environment: minijinja::Environment<'static>,
}

impl ChatTemplate {
    /// Loads and compiles `chat_template.jinja`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TemplateError> {
        let path = path.as_ref();
        let source = std::fs::read_to_string(path).map_err(|source| TemplateError::Read {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_source(&source)
    }

    /// Compiles template text.
    pub fn from_source(source: &str) -> Result<Self, TemplateError> {
        let mut environment = minijinja::Environment::new();
        environment.add_function(
            "raise_exception",
            |message: String| -> Result<Value, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    message,
                ))
            },
        );
        environment.add_function("startswith", |text: String, prefix: String| {
            text.starts_with(&prefix)
        });
        environment.add_function("endswith", |text: String, suffix: String| {
            text.ends_with(&suffix)
        });
        environment.add_filter("tojson", python_tojson);
        environment
            .add_template_owned("chat", shim_template(source))
            .map_err(|error| TemplateError::Compile(error.to_string()))?;
        Ok(Self { environment })
    }

    /// Renders the prompt the model continues.
    pub fn render(&self, context: &RenderContext<'_>) -> Result<String, TemplateError> {
        let template = self
            .environment
            .get_template("chat")
            .map_err(|error| TemplateError::Compile(error.to_string()))?;

        let mut values = serde_json::Map::new();
        values.insert("messages".to_owned(), context.messages.clone());
        values.insert(
            "add_generation_prompt".to_owned(),
            serde_json::Value::Bool(context.add_generation_prompt),
        );
        if let Some(tools) = context.tools {
            values.insert("tools".to_owned(), tools.clone());
        }
        if let Some(enable_thinking) = context.enable_thinking {
            values.insert(
                "enable_thinking".to_owned(),
                serde_json::Value::Bool(enable_thinking),
            );
        }

        template
            .render(Value::from_serialize(&values))
            .map_err(|error| TemplateError::Render(error.to_string()))
    }
}

/// Rewrites the two Python string methods minijinja lacks into function calls.
fn shim_template(source: &str) -> String {
    let rewritten = source
        .replace("content.startswith(", "startswith(content, ")
        .replace("content.endswith(", "endswith(content, ");
    assert!(
        !rewritten.contains(".startswith(") && !rewritten.contains(".endswith("),
        "the shim only knows `content.startswith`/`content.endswith`"
    );
    rewritten
}

/// `jinja2`'s `tojson` filter, reproduced byte for byte.
fn python_tojson(value: Value) -> Result<String, minijinja::Error> {
    let json = serde_json::to_value(&value).map_err(|error| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!("cannot serialise value as json: {error}"),
        )
    })?;
    let mut rendered = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut rendered, PythonJsonFormatter);
    json.serialize(&mut serializer).map_err(|error| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!("cannot serialise value as json: {error}"),
        )
    })?;
    Ok(html_safe_json(
        String::from_utf8_lossy(&rendered).into_owned(),
    ))
}

/// Python's `json.dumps` with `sort_keys=True`: `", "` and `": "` separators and
/// `ensure_ascii=True`.
struct PythonJsonFormatter;

impl serde_json::ser::Formatter for PythonJsonFormatter {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        writer.write_all(b": ")
    }

    fn write_string_fragment<W>(&mut self, writer: &mut W, fragment: &str) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        let mut escaped = String::with_capacity(fragment.len());
        for character in fragment.chars() {
            if character.is_ascii() {
                escaped.push(character);
            } else {
                let mut units = [0_u16; 2];
                for unit in character.encode_utf16(&mut units) {
                    escaped.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
        writer.write_all(escaped.as_bytes())
    }
}

/// `jinja2`'s HTML-safe escaping of a rendered JSON document.
fn html_safe_json(document: String) -> String {
    document
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026")
        .replace('\'', "\\u0027")
}

/// One tool call the model emitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Function name.
    pub name: String,
    /// Arguments as a JSON object string, the way OpenAI reports them.
    pub arguments: String,
}

/// The split of generated text into assistant content and tool calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallParse {
    /// Everything before the first tool call. Text without tool calls is returned
    /// unchanged, so the buffered answer stays byte-identical to the streamed
    /// deltas; text that does lead into a tool call has the separating whitespace
    /// trimmed.
    pub content: String,
    /// Parsed tool calls, in emission order.
    pub tool_calls: Vec<ToolCall>,
}

/// Splits generated text into content and `<tool_call>` blocks.
pub fn parse_tool_calls(text: &str) -> ToolCallParse {
    let mut tool_calls = Vec::new();
    let mut content_end = text.len();
    let mut cursor = 0;

    while let Some(open_offset) = text[cursor..].find(TOOL_CALL_OPEN) {
        let body_start = cursor + open_offset + TOOL_CALL_OPEN.len();
        let Some(close_offset) = text[body_start..].find(TOOL_CALL_CLOSE) else {
            break;
        };
        if tool_calls.is_empty() {
            content_end = cursor + open_offset;
        }
        if let Some(call) = parse_tool_call(&text[body_start..body_start + close_offset]) {
            tool_calls.push(call);
        }
        cursor = body_start + close_offset + TOOL_CALL_CLOSE.len();
    }

    let content = if tool_calls.is_empty() {
        text.to_owned()
    } else {
        text[..content_end].trim_end().to_owned()
    };
    ToolCallParse {
        content,
        tool_calls,
    }
}

/// Parses one `<tool_call>` body: `<function=name>` then `<parameter=k>` blocks.
fn parse_tool_call(body: &str) -> Option<ToolCall> {
    let name_start = body.find(FUNCTION_OPEN)? + FUNCTION_OPEN.len();
    let name_end = body[name_start..].find('>')? + name_start;
    let name = body[name_start..name_end].trim().to_owned();
    let mut body_end = body.len();
    if let Some(offset) = body[name_end..].find(FUNCTION_CLOSE) {
        body_end = name_end + offset;
    }

    let mut arguments = serde_json::Map::new();
    let mut cursor = name_end;
    while let Some(open_offset) = body[cursor..body_end].find(PARAMETER_OPEN) {
        let key_start = cursor + open_offset + PARAMETER_OPEN.len();
        let Some(key_end_offset) = body[key_start..body_end].find('>') else {
            break;
        };
        let key_end = key_start + key_end_offset;
        let key = body[key_start..key_end].trim().to_owned();
        let value_start = key_end + 1;
        let (value, next) = match body[value_start..body_end].find(PARAMETER_CLOSE) {
            Some(close_offset) => {
                let value_end = value_start + close_offset;
                (
                    &body[value_start..value_end],
                    value_end + PARAMETER_CLOSE.len(),
                )
            }
            None => (&body[value_start..body_end], body_end),
        };
        arguments.insert(
            key,
            serde_json::Value::String(
                value
                    .strip_prefix('\n')
                    .unwrap_or(value)
                    .strip_suffix('\n')
                    .unwrap_or(value)
                    .to_owned(),
            ),
        );
        cursor = next.min(body_end);
    }

    Some(ToolCall {
        name,
        arguments: serde_json::Value::Object(arguments).to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn template() -> ChatTemplate {
        ChatTemplate::from_file("tests/fixtures/model/chat_template.jinja")
            .expect("the fixture template is committed")
    }

    fn render(messages: serde_json::Value, tools: Option<serde_json::Value>) -> String {
        let template = template();
        template
            .render(&RenderContext {
                messages: &messages,
                tools: tools.as_ref(),
                add_generation_prompt: true,
                enable_thinking: Some(false),
            })
            .expect("renders")
    }

    #[test]
    fn renders_a_plain_conversation() {
        let text = render(
            json!([
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Hi"}
            ]),
            None,
        );
        assert_eq!(
            text,
            "<|im_start|>system\nYou are terse.<|im_end|>\n\
             <|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn rejects_a_conversation_without_a_user_turn() {
        let template = template();
        let messages = json!([{"role": "assistant", "content": "hi"}]);
        let error = template
            .render(&RenderContext {
                messages: &messages,
                tools: None,
                add_generation_prompt: true,
                enable_thinking: None,
            })
            .expect_err("the template refuses a conversation without a user query");
        assert!(
            error.to_string().contains("No user query found"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn tool_schemas_are_rendered_as_python_json() {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get <weather> & 'more'",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        }]);
        let text = render(
            json!([{"role": "user", "content": "weather?"}]),
            Some(tools),
        );
        let tools_line = text
            .lines()
            .find(|line| line.starts_with("{\"function\""))
            .expect("the tool schema is on its own line");
        assert_eq!(
            tools_line,
            "{\"function\": {\"description\": \"Get \\u003cweather\\u003e \\u0026 \\u0027more\\u0027\", \
             \"name\": \"get_weather\", \"parameters\": {\"properties\": {\"city\": {\"type\": \"string\"}}, \
             \"type\": \"object\"}}, \"type\": \"function\"}"
        );
    }

    #[test]
    fn tool_call_output_is_parsed_back_into_arguments() {
        let text = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n\
                    <parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        let parsed = parse_tool_calls(text);
        assert_eq!(parsed.content, "");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "get_weather");
        let arguments: serde_json::Value =
            serde_json::from_str(&parsed.tool_calls[0].arguments).expect("arguments are json");
        assert_eq!(arguments, json!({"city": "Paris", "days": "3"}));
    }

    #[test]
    fn prose_before_a_tool_call_stays_content() {
        let text = "Let me check.\n<tool_call>\n<function=f>\n</function>\n</tool_call>";
        let parsed = parse_tool_calls(text);
        assert_eq!(parsed.content, "Let me check.");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].arguments, "{}");
    }

    #[test]
    fn plain_text_is_returned_unchanged() {
        let parsed = parse_tool_calls("just an answer\n");
        assert_eq!(parsed.content, "just an answer\n");
        assert!(parsed.tool_calls.is_empty());
    }
}
