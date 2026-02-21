//! # Zantetsu FFI
//!
//! Multi-language FFI bindings for Zantetsu.
//! Exports the core parsing engine to TypeScript/Node (napi-rs),
//! Python (PyO3), and C/C++ (cbindgen).

mod node;

// Re-export for node bindings
pub use zantetsu_core::{HeuristicParser, ParseResult};

#[cfg(feature = "node")]
pub use node::Zantetsu;
