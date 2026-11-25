//! Bounded admission: the backpressure the HTTP layer offers.
//!
//! `max_concurrent_requests` permits are the whole queue. A request that cannot
//! get a permit is rejected with 429 immediately, so a slow engine can never turn
//! into unbounded queuing and unbounded memory.

use std::sync::Arc;

use prometheus_client::metrics::gauge::Gauge;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// The in-flight budget of the server.
pub struct Admission {
    limit: u32,
    permits: Arc<Semaphore>,
    in_flight: Gauge,
}

impl Admission {
    /// Creates `limit` permits; `limit` must be at least one.
    pub fn new(limit: u32, in_flight: Gauge) -> Self {
        assert!(limit >= 1, "max_concurrent_requests must be at least 1");
        Self {
            limit,
            permits: Arc::new(Semaphore::new(limit as usize)),
            in_flight,
        }
    }

    /// The configured limit.
    pub fn limit(&self) -> u32 {
        self.limit
    }

    /// Requests currently holding a permit.
    pub fn in_flight(&self) -> u32 {
        self.limit - self.permits.available_permits() as u32
    }

    /// Takes a permit without waiting; `None` means the server is saturated.
    pub fn try_admit(&self) -> Option<Admitted> {
        let permit = Arc::clone(&self.permits).try_acquire_owned().ok()?;
        self.in_flight.inc();
        Some(Admitted {
            _permit: permit,
            in_flight: self.in_flight.clone(),
        })
    }

    /// Waits until no request holds a permit, `limit` gives up.
    pub async fn wait_idle(&self, limit: std::time::Duration) {
        let deadline = tokio::time::Instant::now() + limit;
        while self.in_flight() > 0 {
            if tokio::time::Instant::now() >= deadline {
                tracing::warn!(
                    in_flight = self.in_flight(),
                    "drain deadline reached with requests still in flight"
                );
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }
}

/// One admission ticket; dropping it frees the permit and updates the gauge.
pub struct Admitted {
    _permit: OwnedSemaphorePermit,
    in_flight: Gauge,
}

impl Drop for Admitted {
    fn drop(&mut self) {
        self.in_flight.dec();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_limit_is_a_hard_cap() {
        let gauge = Gauge::default();
        let admission = Admission::new(1, gauge.clone());
        let first = admission.try_admit().expect("the first request fits");
        assert!(
            admission.try_admit().is_none(),
            "the second must be refused"
        );
        assert_eq!(gauge.get(), 1);
        drop(first);
        assert_eq!(gauge.get(), 0);
        assert!(admission.try_admit().is_some(), "the permit came back");
    }
}
