//! Concrete execution of lowered playspec programs for pysidtracker: an interpreter over a flat
//! encoding of the IR ([`vm`]) and the call machine that runs the play routine of a tune with
//! each modelled entry played as its program ([`machine`]).

pub mod machine;
pub mod vm;

pub use machine::Machine;
pub use vm::{Bail, Vm};
