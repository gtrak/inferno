//! Owning handles over the core's model and sequence C ABI (issue I-036).
//!
//! Zig owns the device memory and decides nothing about its lifetime: it checks
//! a generation counter when it is called. This module owns the *policy*, and it
//! does that in the type system rather than with a runtime guard:
//!
//! * [`Model`] and [`Sequence`] close their handle in `Drop`, so no handle can
//!   be forgotten on an error path;
//! * [`Sequence`] carries the lifetime of the model it was opened on, so a
//!   sequence that outlives its model does not compile (`Model::close` takes
//!   `self` by value and cannot be called while a sequence borrows it). The C
//!   ABI cannot forbid that shape; it is unreachable from here;
//! * a handle-taking method takes `&mut self` and `close` consumes `self`, so a
//!   closed handle is not in scope to step;
//! * both types are `!Send` and `!Sync`, which is the honest statement of the
//!   core's contract: its CUDA context is single-context, one thread's at a
//!   time. Moving a handle across a thread is a compile error, not a race that
//!   the core's runtime checks may or may not catch.
//!
//! Every refusal the core reports becomes a [`HandleError`] here; no call is
//! silent and no error is dropped. The numbers in [`Code`] are the C ABI's, and
//! `core/src/model/handles.zig` declares them — a test on each side pins the
//! numbers, so the two declarations cannot drift without a red suite.

use std::ffi::c_char;
use std::marker::PhantomData;

/// Floats the core writes to `out_stats` in one step.
///
/// Mirrors `stats_len` in `core/src/model/handles.zig`: generated ids, tokens in
/// the caches, and the step's wall time in microseconds.
pub const STATS_LEN: usize = 3;

/// An open model, owned by the core. The type is opaque on purpose: Rust never
/// looks inside a handle, it only passes it back.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct inferno_model {
    _private: [u8; 0],
}

/// An open sequence, owned by the core.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct inferno_seq {
    _private: [u8; 0],
}

/// Counters of the core's handle tables (the C ABI's `inferno_handle_counts`).
///
/// `leaked_models` is the core's own allocator reporting, when a model closes,
/// that an allocation it handed out was never returned.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HandleCounts {
    /// Models currently open.
    pub models_live: u32,
    /// Models opened since process start.
    pub models_opened: u32,
    /// Sequences currently open.
    pub sequences_live: u32,
    /// Sequences opened since process start.
    pub sequences_opened: u32,
    /// Sequences whose model was closed and which have not been dropped yet.
    pub sequences_orphaned: u32,
    /// Models that closed with an allocation outstanding.
    pub leaked_models: u32,
}

unsafe extern "C" {
    fn inferno_model_open(manifest_json: *const c_char, len: u32) -> *mut inferno_model;
    fn inferno_model_close(model: *mut inferno_model);
    fn inferno_seq_open(
        model: *const inferno_model,
        ids: *const u32,
        count: u32,
    ) -> *mut inferno_seq;
    fn inferno_seq_close(sequence: *mut inferno_seq);
    fn inferno_seq_reset(sequence: *const inferno_seq) -> i32;
    fn inferno_seq_step(
        sequence: *const inferno_seq,
        out_token: *mut u32,
        out_stats: *mut f32,
        stats_cap: u32,
    ) -> i32;
    fn inferno_handle_counts(out: *mut HandleCounts) -> i32;
}

/// The codes the core returns, exactly as `handles.Code` declares them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Code {
    /// The call did what it says.
    Ok = 0,
    /// A null argument, an out-of-range id, or a zero-length document.
    InvalidArgument = -1,
    /// Every slot of the core's handle table is taken.
    OutOfHandles = -2,
    /// The handle names no live object: closed, or its slot was reissued.
    StaleHandle = -3,
    /// The model the sequence belongs to has been closed.
    ModelClosed = -4,
    /// The model could not be opened: no device, or a failed load.
    ModelUnavailable = -5,
    /// The manifest names a model the core cannot run.
    UnsupportedModel = -6,
    /// The prompt was empty.
    EmptyPrompt = -7,
    /// The prompt does not fit one prefill.
    PromptTooLong = -8,
    /// The sequence reached the end of the model's arena.
    ContextFull = -9,
    /// The forward failed for a reason without a code of its own.
    StepFailed = -10,
}

impl Code {
    /// The code `value` names, or `None` when the core returned a number this
    /// module does not know (a drifted ABI, which is a build to fix).
    pub fn from_raw(value: i32) -> Option<Code> {
        let code = match value {
            0 => Code::Ok,
            -1 => Code::InvalidArgument,
            -2 => Code::OutOfHandles,
            -3 => Code::StaleHandle,
            -4 => Code::ModelClosed,
            -5 => Code::ModelUnavailable,
            -6 => Code::UnsupportedModel,
            -7 => Code::EmptyPrompt,
            -8 => Code::PromptTooLong,
            -9 => Code::ContextFull,
            -10 => Code::StepFailed,
            _ => return None,
        };
        Some(code)
    }
}

/// A handle call the core refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum HandleError {
    /// An open returned null. The C ABI's open entry points have no error
    /// out-parameter, so the reason reaches the operator through the core's log.
    #[error("the core refused to open the handle; its log carries the code")]
    OpenRefused,
    /// A documented [`Code`] from an entry point that returns one.
    #[error("the core refused with {0:?}")]
    Code(Code),
    /// A number this module does not know: the ABI has drifted.
    #[error("the core returned the unknown handle code {0}")]
    Unknown(i32),
}

/// Turns a returned code into `Result<(), HandleError>`.
fn accepted(code: i32) -> Result<(), HandleError> {
    match Code::from_raw(code) {
        Some(Code::Ok) => Ok(()),
        Some(other) => Err(HandleError::Code(other)),
        None => Err(HandleError::Unknown(code)),
    }
}

/// The counters a step reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    /// Ids the sequence produced since it was opened or reset.
    pub generated: u32,
    /// Tokens in the caches: the prompt plus every id fed back.
    pub cached: u32,
    /// Wall time of this step.
    pub step_micros: f32,
}

/// One decode step: the id, and the core's counters for it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Step {
    /// The greedy id this step produced. It is not in the caches yet.
    pub token: u32,
    /// The counters the same call reported.
    pub stats: Stats,
}

/// A model the core has open.
///
/// Closing is `Drop`'s job, so a handle that is dropped is a handle that is
/// closed; [`Model::close`] exists to say so at the call site and to consume the
/// value, which is what stops a closed model from being used again.
///
/// A model handle is single-context, and `!Send` is how that is stated —
/// sending one to another thread must not compile:
///
/// ```compile_fail
/// use inferno_server::engine::handles::Model;
/// fn require_send<T: Send>(_: T) {}
/// let model: Model = unsafe { std::mem::zeroed() };
/// require_send(model); // error[E0277]: `*mut inferno_model` cannot be sent
/// ```
pub struct Model {
    raw: *mut inferno_model,
}

impl Model {
    /// The committed tiny fixture: the only forward the core has until I-017c
    /// lands the real checkpoint.
    pub const TINY_MANIFEST: &'static str = "{\"model\":\"tiny\"}";

    /// Opens the model `manifest_json` describes.
    ///
    /// The document is a JSON object with a `model` key. `{"model":"tiny"}` is
    /// the committed tiny fixture; the real checkpoint is refused until the
    /// forward for it exists, so a caller cannot believe it opened something the
    /// core cannot run.
    pub fn open(manifest_json: &str) -> Result<Model, HandleError> {
        let length = u32::try_from(manifest_json.len())
            .map_err(|_| HandleError::Code(Code::InvalidArgument))?;
        // SAFETY: the document is alive for the call and the core copies no
        // pointer out of it; the returned handle is owned by `Model`.
        let raw = unsafe { inferno_model_open(manifest_json.as_ptr().cast(), length) };
        if raw.is_null() {
            Err(HandleError::OpenRefused)
        } else {
            Ok(Model { raw })
        }
    }

    /// Opens the committed tiny fixture.
    pub fn open_tiny() -> Result<Model, HandleError> {
        Model::open(Self::TINY_MANIFEST)
    }

    /// Opens a sequence over this model and prefills `ids`.
    ///
    /// The returned sequence borrows the model for `'m`, which is the whole
    /// point: `Model::close` takes `self`, so it cannot run while a sequence is
    /// live, and the borrow checker is what enforces that (not a runtime check
    /// in the core).
    pub fn sequence<'m>(&'m mut self, ids: &[u32]) -> Result<Sequence<'m>, HandleError> {
        let count =
            u32::try_from(ids.len()).map_err(|_| HandleError::Code(Code::InvalidArgument))?;
        // SAFETY: `self.raw` is a live model handle, `ids` is alive for the
        // call, and the sequence keeps the model borrowed for `'m`.
        let raw = unsafe { inferno_seq_open(self.raw, ids.as_ptr(), count) };
        if raw.is_null() {
            Err(HandleError::OpenRefused)
        } else {
            Ok(Sequence {
                raw,
                model: PhantomData,
            })
        }
    }

    /// Closes the model. Consumes `self`, so nothing can step it afterwards.
    pub fn close(mut self) {
        self.release();
    }

    /// Calls the core's close once; `Drop` and `close` both land here.
    fn release(&mut self) {
        let raw = std::mem::replace(&mut self.raw, std::ptr::null_mut());
        if raw.is_null() {
            // Already closed through the raw ABI or by `close`: nothing to do.
        } else {
            // SAFETY: `raw` came from `inferno_model_open` and is closed once.
            unsafe { inferno_model_close(raw) };
        }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        self.release();
    }
}

/// A sequence bound to the model it was opened on.
///
/// The lifetime parameter is what makes "a sequence outlives its model"
/// unrepresentable:
///
/// ```compile_fail
/// use inferno_server::engine::handles::Model;
/// let mut model: Model = unsafe { std::mem::zeroed() };
/// let mut sequence = model.sequence(&[1u32]).expect("open");
/// model.close(); // error[E0505]: cannot move out of `model` because it is borrowed
/// let _ = sequence.step().is_ok();
/// ```
pub struct Sequence<'m> {
    raw: *mut inferno_seq,
    model: PhantomData<&'m mut Model>,
}

impl Sequence<'_> {
    /// Produces the next greedy id, extending the caches by one token.
    pub fn step(&mut self) -> Result<Step, HandleError> {
        let mut token = 0u32;
        let mut stats = [0.0f32; STATS_LEN];
        // SAFETY: the handle is live (this method has `&mut self`), the two
        // out-parameters are valid for the call and sized by `STATS_LEN`.
        let code = unsafe {
            inferno_seq_step(
                self.raw,
                &mut token,
                stats.as_mut_ptr(),
                u32::try_from(STATS_LEN).expect("STATS_LEN fits in u32"),
            )
        };
        accepted(code)?;
        Ok(Step {
            token,
            stats: Stats {
                generated: stats[0] as u32,
                cached: stats[1] as u32,
                step_micros: stats[2],
            },
        })
    }

    /// Returns the sequence to start-of-sequence and replays its prompt.
    ///
    /// The core keeps the prompt it was opened with, so a reset is a replay and
    /// the same ids come back — that is the property the core's reset contract
    /// is tested against.
    pub fn reset(&mut self) -> Result<(), HandleError> {
        // SAFETY: the handle is live (this method has `&mut self`).
        let code = unsafe { inferno_seq_reset(self.raw) };
        accepted(code)
    }

    /// Closes the sequence. Consumes `self`, so it cannot be stepped afterwards:
    ///
    /// ```compile_fail
    /// use inferno_server::engine::handles::{Model, Sequence};
    /// let mut model: Model = unsafe { std::mem::zeroed() };
    /// let mut sequence: Sequence<'_> = model.sequence(&[1u32]).expect("open");
    /// sequence.close();
    /// sequence.step(); // error[E0382]: use of moved value: `sequence`
    /// ```
    pub fn close(mut self) {
        self.release();
    }

    /// Calls the core's close once; `Drop` and `close` both land here.
    fn release(&mut self) {
        let raw = std::mem::replace(&mut self.raw, std::ptr::null_mut());
        if raw.is_null() {
            // Already closed: nothing to do.
        } else {
            // SAFETY: `raw` came from `inferno_seq_open` and is closed once.
            unsafe { inferno_seq_close(raw) };
        }
    }
}

impl Drop for Sequence<'_> {
    fn drop(&mut self) {
        self.release();
    }
}

/// The core's handle-table counters.
///
/// This is the leak check the Rust side can make: after a `Sequence` is dropped
/// and a `Model` is closed, every counter that is not an `_opened` total is
/// zero. `leaked_models` is the core's allocator agreeing.
pub fn handle_counts() -> Result<HandleCounts, HandleError> {
    let mut counts = HandleCounts::default();
    // SAFETY: the pointer is valid for one `HandleCounts` write.
    let code = unsafe { inferno_handle_counts(&mut counts) };
    accepted(code)?;
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed fixture's prompt (`tests/golden/tiny/greedy_ids.json`).
    const PROMPT: [u32; 12] = [13, 4, 199, 42, 7, 88, 211, 30, 96, 155, 61, 240];

    /// The first two ids of the architecture gate, byte for byte: the same 16
    /// ids the Zig suite asserts, taken one at a time through the C ABI.
    const GATE_IDS: [u32; 2] = [169, 252];

    #[test]
    fn the_codes_are_the_numbers_the_zig_abi_declares() {
        assert_eq!(0, Code::Ok as i32);
        assert_eq!(-1, Code::InvalidArgument as i32);
        assert_eq!(-2, Code::OutOfHandles as i32);
        assert_eq!(-3, Code::StaleHandle as i32);
        assert_eq!(-4, Code::ModelClosed as i32);
        assert_eq!(-5, Code::ModelUnavailable as i32);
        assert_eq!(-6, Code::UnsupportedModel as i32);
        assert_eq!(-7, Code::EmptyPrompt as i32);
        assert_eq!(-8, Code::PromptTooLong as i32);
        assert_eq!(-9, Code::ContextFull as i32);
        assert_eq!(-10, Code::StepFailed as i32);
        for value in [0i32, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10] {
            assert!(Code::from_raw(value).is_some(), "{value} is a code");
        }
        assert_eq!(None, Code::from_raw(-11));
        assert_eq!(None, Code::from_raw(1));
    }

    #[test]
    fn null_and_forged_handles_are_refused_with_a_code_not_a_crash() {
        let mut token = 0u32;
        let code =
            unsafe { inferno_seq_step(std::ptr::null_mut(), &mut token, std::ptr::null_mut(), 0) };
        assert_eq!(Some(Code::InvalidArgument), Code::from_raw(code));
        let code = unsafe { inferno_seq_reset(std::ptr::null_mut()) };
        assert_eq!(Some(Code::InvalidArgument), Code::from_raw(code));
        let raw = unsafe { inferno_seq_open(std::ptr::null_mut(), PROMPT.as_ptr(), 12) };
        assert!(raw.is_null());
        let raw = unsafe { inferno_model_open(std::ptr::null(), 0) };
        assert!(raw.is_null());
        let code = unsafe { inferno_handle_counts(std::ptr::null_mut()) };
        assert_eq!(Some(Code::InvalidArgument), Code::from_raw(code));

        // A pointer the core never issued must be refused by its address test,
        // not dereferenced. `dangling_mut` is aligned and definitely not a cell.
        let forged = std::ptr::dangling_mut::<inferno_seq>();
        let code = unsafe { inferno_seq_step(forged, &mut token, std::ptr::null_mut(), 0) };
        assert_eq!(Some(Code::StaleHandle), Code::from_raw(code));
        let code = unsafe { inferno_seq_reset(forged) };
        assert_eq!(Some(Code::StaleHandle), Code::from_raw(code));
        // Closing a forged handle is a no-op, never a free of foreign memory.
        unsafe { inferno_seq_close(forged) };
        unsafe { inferno_model_close(forged.cast()) };
        assert_eq!(0, unsafe {
            inferno_handle_counts(&mut HandleCounts::default())
        });
    }

    /// The one test that touches a GPU: the tiny fixture is the only forward the
    /// core has, and a model open is a device context. Everything that needs a
    /// model lives here so `cargo test` never runs two contexts on one device —
    /// the concurrency that is measured to flake.
    #[test]
    fn a_sequence_is_opened_stepped_reset_and_dropped_without_leaking() {
        let before = handle_counts().expect("the core answers its counters");
        let mut model = Model::open_tiny().expect("the tiny fixture model opens");

        {
            let mut sequence = model
                .sequence(&PROMPT)
                .expect("the committed prompt prefills");
            let first = sequence.step().expect("the first id");
            assert_eq!(GATE_IDS[0], first.token);
            assert_eq!(1, first.stats.generated);
            assert_eq!(PROMPT.len() as u32, first.stats.cached);
            assert!(first.stats.step_micros >= 0.0);
            let second = sequence.step().expect("the second id");
            assert_eq!(GATE_IDS[1], second.token);
            assert_eq!(2, second.stats.generated);
            sequence.reset().expect("reset replays the prompt");
            assert_eq!(
                GATE_IDS[0],
                sequence.step().expect("the replayed first id").token
            );
            // Dropped here: the sequence handle is closed by `Drop`.
        }
        let dropped = handle_counts().expect("counters");
        assert_eq!(0, dropped.sequences_live, "a dropped sequence is closed");
        assert_eq!(0, dropped.sequences_orphaned);

        // The C ABI's refusals, driven from Rust with raw handles: a closed
        // sequence, and a sequence whose model was closed under it.
        let raw_model = model.raw;
        let raw_sequence =
            unsafe { inferno_seq_open(raw_model, PROMPT.as_ptr(), PROMPT.len() as u32) };
        assert!(!raw_sequence.is_null());
        let mut token = 0u32;
        let code = unsafe { inferno_seq_step(raw_sequence, &mut token, std::ptr::null_mut(), 0) };
        assert_eq!(Some(Code::Ok), Code::from_raw(code));
        unsafe { inferno_seq_close(raw_sequence) };
        let code = unsafe { inferno_seq_step(raw_sequence, &mut token, std::ptr::null_mut(), 0) };
        assert_eq!(Some(Code::StaleHandle), Code::from_raw(code));

        let raw_orphan =
            unsafe { inferno_seq_open(raw_model, PROMPT.as_ptr(), PROMPT.len() as u32) };
        assert!(!raw_orphan.is_null());
        unsafe { inferno_model_close(raw_model) };
        let code = unsafe { inferno_seq_step(raw_orphan, &mut token, std::ptr::null_mut(), 0) };
        assert_eq!(Some(Code::ModelClosed), Code::from_raw(code));
        let code = unsafe { inferno_seq_reset(raw_orphan) };
        assert_eq!(Some(Code::ModelClosed), Code::from_raw(code));
        unsafe { inferno_seq_close(raw_orphan) };

        // The model was closed through the raw ABI above, so `Model::close` and
        // `Drop` are now no-ops: closing twice is defined, not a double free.
        model.close();
        let after = handle_counts().expect("counters");
        assert_eq!(0, after.models_live);
        assert_eq!(0, after.sequences_live);
        assert_eq!(0, after.sequences_orphaned);
        assert_eq!(0, after.leaked_models, "the model freed every allocation");
        assert_eq!(before.models_opened + 1, after.models_opened);
    }
}
