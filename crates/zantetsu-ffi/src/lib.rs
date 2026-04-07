//! # Zantetsu FFI
//!
//! Multi-language FFI bindings for Zantetsu.
//! Exports the core parsing engine to TypeScript/Node (napi-rs),
//! Python (PyO3), and C/C++ (cbindgen).
//!
//! Crates:
//! - [`zantetsu`](https://docs.rs/zantetsu) - unified API surface
//! - [`zantetsu-core`](https://docs.rs/zantetsu-core) - parsing engine
//! - [`zantetsu-vecdb`](https://docs.rs/zantetsu-vecdb) - canonical title matching
//! - [`zantetsu-trainer`](https://docs.rs/zantetsu-trainer) - training workflows
//! - [`zantetsu-ffi`](https://docs.rs/zantetsu-ffi) - Node/Python/C bindings
//!
//! Enable `node` or `python` features depending on the target binding surface you need.

#[cfg(feature = "node")]
mod node;

#[cfg(feature = "python")]
mod python;

// Re-export for node bindings
pub use zantetsu_core::{HeuristicParser, ParseResult};

#[cfg(feature = "node")]
pub use node::HeuristicParserNode;

#[cfg(feature = "python")]
pub use python::HeuristicParserPy;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn zantetsu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::pymodule(m)
}
