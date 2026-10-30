//! A 3x3x3 Rubik's cube engine with bandaging support and a meet-in-the-middle search for
//! short algorithms (3-cycles, flips, twists).
//!
//! The Python bindings live in [`python`] behind the `python` feature.

pub mod cube;
pub mod geometry;
pub mod moves;
pub mod search;

#[cfg(feature = "python")]
mod python;

pub use cube::{Bandage, Cube, Effect, State};
pub use moves::{format_alg, invert, parse_alg, parse_moveset, simplify};
pub use search::{Found, Kind, Query, SearchStats, Solver};
