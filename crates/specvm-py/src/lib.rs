//! `pysidtracker._specvm`: the native call machine for Python.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyByteArray;
use specvm_core::machine::SID_REG_COUNT;
use specvm_core::vm::{Ins, Op, NAMES};
use specvm_core::{Bail, Machine, Vm};

const OK: u8 = 0;
const NEED: u8 = 1;
const BAIL: u8 = 2;

fn invalid(message: impl ToString) -> PyErr {
    PyValueError::new_err(message.to_string())
}

/// The programs of one tune and the 6502 machine that calls them.
#[pyclass(unsendable)]
struct Core {
    machine: Machine,
    _ram: PyBuffer<u8>,
}

#[pymethods]
impl Core {
    /// `ram` is the writable 64 KiB the tune runs in, shared and not copied; the play routine
    /// is at `play` and `entry` is the modelled entry it is or reaches.
    #[new]
    fn new(ram: &Bound<'_, PyAny>, play: u16, entry: u16, max_cycles: u64) -> PyResult<Self> {
        let buffer = PyBuffer::<u8>::get(ram)?;
        if buffer.readonly() || buffer.item_count() != 0x10000 || !buffer.is_c_contiguous() {
            return Err(invalid("the memory is not a writable contiguous 64 KiB"));
        }
        let ptr = buffer.buf_ptr().cast::<u8>();
        // SAFETY: the buffer is held for the life of the machine and its 64 KiB is written by
        // nothing else while a frame runs, which happens with the GIL held.
        let machine = unsafe { Machine::new(ptr, play, entry, max_cycles) };
        Ok(Core {
            machine,
            _ram: buffer,
        })
    }

    /// Adds the memory map under port setting `bits`: whether I/O is mapped, the 4 KiB ROM
    /// `pages` as `(page, image)` and the modelled `entries`.
    fn add_bank(
        &mut self,
        bits: usize,
        io: bool,
        pages: Vec<(u8, Vec<u8>)>,
        entries: Vec<u16>,
    ) -> PyResult<()> {
        self.machine
            .add_bank(bits, io, pages, &entries)
            .map_err(invalid)
    }

    /// Maps the memory under port setting `bits`.
    fn select(&mut self, bits: usize) -> PyResult<()> {
        self.machine.select(bits).map_err(invalid)
    }

    /// The port setting the memory is mapped under.
    #[getter]
    fn bits(&self) -> usize {
        self.machine.bits()
    }

    /// Adds the program of modelled entry `entry` under port setting `bits`: `code` is
    /// little-endian `(op: u64, arg: i64)` pairs, `inputs` the `(variable, register)` of the
    /// entry registers it reads.
    #[pyo3(signature = (bits, entry, code, tables, inputs, variables, controls))]
    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        bits: usize,
        entry: u16,
        code: &[u8],
        tables: Vec<Vec<i64>>,
        inputs: Vec<(usize, usize)>,
        variables: usize,
        controls: usize,
    ) -> PyResult<()> {
        let (pairs, rest) = code.as_chunks::<16>();
        if !rest.is_empty() {
            return Err(invalid("the code is not whole instructions"));
        }
        let mut ins = Vec::with_capacity(pairs.len());
        for pair in pairs {
            let op = u64::from_le_bytes(pair[..8].try_into().unwrap());
            let arg = i64::from_le_bytes(pair[8..].try_into().unwrap());
            let op = u16::try_from(op)
                .ok()
                .and_then(Op::from_u16)
                .ok_or_else(|| invalid(format!("no opcode {op}")))?;
            ins.push(Ins { op, arg });
        }
        let vm = Vm::new(ins, tables, inputs, variables, controls).map_err(invalid)?;
        self.machine.add(bits, entry, vm).map_err(invalid)
    }

    /// Sets the CPU: `(pc, sp, a, x, y, p, cycles)`.
    fn set_cpu(&mut self, state: (u16, u8, u8, u8, u8, u8, u64)) {
        let cpu = &mut self.machine.cpu;
        (cpu.pc, cpu.sp, cpu.a, cpu.x, cpu.y) = (state.0, state.1, state.2, state.3, state.4);
        cpu.set_p(state.5);
        cpu.cycles = state.6;
    }

    /// The CPU: `(pc, sp, a, x, y, p, cycles)`.
    fn cpu(&self) -> (u16, u8, u8, u8, u8, u8, u64) {
        let c = &self.machine.cpu;
        (c.pc, c.sp, c.a, c.x, c.y, c.p(), c.cycles)
    }

    /// The control states of the program of `entry` under port setting `bits`.
    fn controls(&mut self, bits: usize, entry: u16) -> PyResult<Vec<i64>> {
        let vm = self
            .machine
            .vm(bits, entry)
            .ok_or_else(|| invalid("no such program"))?;
        Ok(vm.controls().to_vec())
    }

    fn set_controls(&mut self, bits: usize, entry: u16, values: Vec<i64>) -> PyResult<()> {
        let vm = self
            .machine
            .vm(bits, entry)
            .ok_or_else(|| invalid("no such program"))?;
        vm.set_controls(&values).map_err(invalid)
    }

    /// Plays up to `frames` frames, writing each one's 25 SID registers to `out` from frame
    /// `first`: `(frames played, status, entry)` where status 0 is done, 1 a program for `entry`
    /// is needed and 2 that Python must play the next frame.
    fn run(
        &mut self,
        frames: usize,
        out: &Bound<'_, PyByteArray>,
        first: usize,
    ) -> PyResult<(usize, u8, u16)> {
        if out.len() < (first + frames) * SID_REG_COUNT {
            return Err(PyRuntimeError::new_err("the output is too small"));
        }
        // SAFETY: the GIL is held, so nothing resizes or reads the array during the run.
        let bytes = unsafe { out.as_bytes_mut() };
        let (done, why) = self
            .machine
            .run(frames, &mut bytes[first * SID_REG_COUNT..]);
        Ok(match why {
            None => (done, OK, 0),
            Some(Bail::Need(entry)) => (done, NEED, entry),
            Some(Bail::Bank(bits)) => (done, NEED, bits as u16),
            Some(Bail::Device(addr)) => (done, BAIL, addr),
            Some(why) => (done, BAIL, 0x100 + why.code()),
        })
    }

    /// The SID writes the programs made in the last frame played, `(register, value)`.
    fn writes(&self) -> Vec<(u8, u8)> {
        self.machine.sid.clone()
    }
}

/// The opcode numbering the encoder shares.
#[pyfunction]
fn opcodes() -> Vec<&'static str> {
    NAMES.to_vec()
}

#[pymodule]
fn _specvm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Core>()?;
    m.add_function(wrap_pyfunction!(opcodes, m)?)?;
    Ok(())
}
