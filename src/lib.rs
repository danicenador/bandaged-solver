//! A 3x3x3 and 4x4x4 Rubik's cube engine with bandaging support and a meet-in-the-middle search for
//! short algorithms (3-cycles, flips, twists), plus a difficulty study of bandaged cubes ([`study`]).
//!
//! The Python bindings live in [`python`] behind the `python` feature.

pub mod cube;
pub mod cube4;
pub mod geometry;
pub mod group;
pub mod human;
pub mod moves;
pub mod search;
pub mod study;

#[cfg(feature = "python")]
mod python;

pub use cube::{Bandage, Cube, Effect, State};
pub use cube4::{Bandage4, Cube4, State4};
pub use moves::{format_alg, invert, parse_alg, parse_moveset, simplify};
pub use search::{Found, Kind, Puzzle, Query, SearchStats, Solver};
