//! [`ZigEngine`]: the C ABI of the Zig core behind the [`Engine`] trait.
//!
//! The core calls [`route_token`] from a Zig worker thread with nothing but the
//! request id, so `submit` registers the request's channel in [`TokenRoutes`]
//! before handing the request over and the callback looks it up. The callback
//! only pushes into that channel: no core lock is held while Rust runs, and a
//! panic inside it is caught before it could unwind across the FFI boundary.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;

use super::{Engine, EngineError, EngineToken, GenerateRequest, TokenSink, ffi};

/// Capacity of the metrics document buffer (`inferno_core_metrics`).
const METRICS_BUFFER_BYTES: u32 = 64 * 1024;

/// The engine handle: an owned pointer the core handed out in `init`.
pub struct ZigEngine {
    /// Null once `shutdown` ran; the core must not be called after that.
    handle: AtomicPtr<ffi::inferno_engine>,
    /// Address of the [`TokenRoutes`] the callback context points at.
    context: AtomicPtr<TokenRoutes>,
    /// Strong reference to the same routes, for lookups from `submit`/`cancel`.
    routes: Arc<TokenRoutes>,
    version: String,
}

impl ZigEngine {
    /// Initialises the core with `config_json` and installs the token callback.
    pub fn initialise(config_json: &str) -> Result<Self, EngineError> {
        if !ffi::CORE_ABI_LINKED {
            return Err(EngineError::Failed(
                "the linked core does not export the engine ABI yet (issue I-15); run with --engine mock"
                    .to_owned(),
            ));
        }
        let document = CString::new(config_json).map_err(|error| {
            EngineError::Failed(format!("core config is not a C string: {error}"))
        })?;
        let length = u32::try_from(document.as_bytes().len())
            .map_err(|_| EngineError::Failed("core config is too long".to_owned()))?;

        // SAFETY: `document` is a NUL terminated string valid for `length` bytes and
        // `inferno_core_init` only reads them.
        let handle = unsafe { ffi::inferno_core_init(document.as_ptr(), length) };
        if handle.is_null() {
            return Err(EngineError::NotInitialised);
        }

        let routes = Arc::new(TokenRoutes::default());
        let context = Arc::into_raw(Arc::clone(&routes)).cast_mut();
        // SAFETY: the handle is live and `context` stays valid until `shutdown` has
        // returned (it owns one strong reference to the routes).
        let version = unsafe {
            ffi::inferno_core_set_token_cb(handle, Some(route_token), context.cast::<c_void>());
            CStr::from_ptr(ffi::inferno_core_version())
                .to_string_lossy()
                .into_owned()
        };

        Ok(Self {
            handle: AtomicPtr::new(handle),
            context: AtomicPtr::new(context),
            routes,
            version,
        })
    }

    /// True once the core handle exists.
    fn is_live(&self) -> bool {
        !self.handle.load(Ordering::Acquire).is_null()
    }

    /// Registers the channel the callback pushes into.
    fn register(&self, req_id: u64, sink: TokenSink) {
        self.routes.insert(req_id, sink);
    }

    /// Forgets a request's channel; the callback then drops its tokens.
    fn forget(&self, req_id: u64) {
        self.routes.remove(req_id);
    }
}

impl Engine for ZigEngine {
    fn version(&self) -> String {
        self.version.clone()
    }

    fn submit(&self, req_id: u64, req: GenerateRequest) -> Result<(), EngineError> {
        let handle = self.handle.load(Ordering::Acquire);
        if handle.is_null() {
            return Err(EngineError::NotInitialised);
        }
        let document = CString::new(req.wire_json()).map_err(|error| {
            EngineError::Failed(format!("request document is not a C string: {error}"))
        })?;
        let length = u32::try_from(document.as_bytes().len())
            .map_err(|_| EngineError::Failed("request document is too long".to_owned()))?;

        self.register(req_id, req.sink.clone());
        // SAFETY: the handle is live (checked above) and the document outlives the call.
        let code = unsafe { ffi::inferno_core_submit(handle, req_id, document.as_ptr(), length) };
        if code < 0 {
            self.forget(req_id);
            return Err(EngineError::Rejected(code));
        }
        Ok(())
    }

    fn cancel(&self, req_id: u64) -> Result<(), EngineError> {
        self.forget(req_id);
        let handle = self.load_handle()?;
        // SAFETY: the handle is live (checked above).
        let code = unsafe { ffi::inferno_core_cancel(handle, req_id) };
        if code < 0 {
            return Err(EngineError::Rejected(code));
        }
        Ok(())
    }

    fn metrics_json(&self) -> String {
        let Ok(handle) = self.load_handle() else {
            return "{}".to_owned();
        };
        let mut buffer = vec![0_u8; METRICS_BUFFER_BYTES as usize];
        // SAFETY: `buffer` is writable for `cap` bytes and the core writes at most that many.
        let written = unsafe {
            ffi::inferno_core_metrics(
                handle,
                buffer.as_mut_ptr().cast::<c_char>(),
                METRICS_BUFFER_BYTES,
            )
        };
        if written <= 0 {
            return "{}".to_owned();
        }
        let written = usize::try_from(written).unwrap_or(0).min(buffer.len());
        String::from_utf8_lossy(&buffer[..written]).into_owned()
    }

    fn ready(&self) -> bool {
        self.is_live()
    }

    fn shutdown(&self) {
        let handle = self.handle.swap(std::ptr::null_mut(), Ordering::AcqRel);
        if handle.is_null() {
            return;
        }
        // SAFETY: the handle came from `inferno_core_init` and is shut down exactly once.
        unsafe { ffi::inferno_core_shutdown(handle) };
        self.routes.clear();
        let context = self.context.swap(std::ptr::null_mut(), Ordering::AcqRel);
        if !context.is_null() {
            // SAFETY: `context` is the pointer `Arc::into_raw` handed out in `initialise`;
            // the core no longer runs callbacks once `shutdown` returned.
            drop(unsafe { Arc::from_raw(context) });
        }
    }
}

impl ZigEngine {
    /// The live core handle, or the error the HTTP layer turns into a 503.
    fn load_handle(&self) -> Result<*mut ffi::inferno_engine, EngineError> {
        let handle = self.handle.load(Ordering::Acquire);
        if handle.is_null() {
            return Err(EngineError::NotInitialised);
        }
        Ok(handle)
    }
}

impl Drop for ZigEngine {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Request id → token channel, shared with the core callback.
#[derive(Default)]
struct TokenRoutes {
    sinks: Mutex<HashMap<u64, TokenSink>>,
}

impl TokenRoutes {
    /// Publishes the channel of `req_id`.
    fn insert(&self, req_id: u64, sink: TokenSink) {
        if let Ok(mut sinks) = self.sinks.lock() {
            sinks.insert(req_id, sink);
        }
    }

    /// Drops the channel of `req_id`.
    fn remove(&self, req_id: u64) {
        if let Ok(mut sinks) = self.sinks.lock() {
            sinks.remove(&req_id);
        }
    }

    /// Drops every channel, so a callback racing shutdown finds nothing.
    fn clear(&self) {
        if let Ok(mut sinks) = self.sinks.lock() {
            sinks.clear();
        }
    }

    /// The channel of `req_id`, if the request is still in flight.
    fn sink_of(&self, req_id: u64) -> Option<TokenSink> {
        self.sinks
            .lock()
            .ok()
            .and_then(|sinks| sinks.get(&req_id).cloned())
    }
}

/// The token callback the core calls from a Zig worker thread.
///
/// # Safety
///
/// `context` must be the `Arc<TokenRoutes>` pointer installed by
/// [`ZigEngine::initialise`] and still alive, which holds until `shutdown` has
/// returned.
unsafe extern "C" fn route_token(context: *mut c_void, req_id: u64, token_id: u32, flags: u32) {
    if context.is_null() {
        return;
    }
    // SAFETY: `context` is the routes pointer, alive for the whole engine lifetime.
    let routes = unsafe { &*context.cast::<TokenRoutes>() };
    let Some(sink) = routes.sink_of(req_id) else {
        return;
    };
    let token = EngineToken {
        id: token_id,
        flags,
    };

    match sink.try_send(token) {
        Ok(()) => {}
        Err(mpsc::error::TrySendError::Full(token)) => {
            // The consumer is behind: block this worker thread rather than drop a
            // token. The request's stream drains continuously, so this is rare, and a
            // panic here (runtime shutting down) must not cross the FFI boundary.
            let fallback = sink.clone();
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                let _ = fallback.blocking_send(token);
            }));
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_forget_removed_requests() {
        let routes = TokenRoutes::default();
        let (sink, mut receiver) = mpsc::channel(4);
        routes.insert(7, sink);
        assert!(routes.sink_of(7).is_some());
        routes.remove(7);
        assert!(routes.sink_of(7).is_none());
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn callback_delivers_to_the_registered_channel() {
        let routes = Arc::new(TokenRoutes::default());
        let (sink, mut receiver) = mpsc::channel(4);
        routes.insert(11, sink);
        let context = Arc::into_raw(Arc::clone(&routes))
            .cast_mut()
            .cast::<c_void>();

        // SAFETY: `context` points at the routes, which the local `routes` keeps alive.
        unsafe { route_token(context, 11, 42, super::super::TOKEN_FLAG_FINAL) };

        let token = receiver.try_recv().expect("the callback delivered a token");
        assert_eq!(token.id, 42);
        assert!(token.is_final());
        assert!(!token.is_error());

        // SAFETY: the pointer came from `Arc::into_raw` above and is reclaimed once.
        drop(unsafe { Arc::from_raw(context.cast::<TokenRoutes>()) });
    }
}
