//! Event-driven objects: counters, owners, accumulators and cursors, stepped natively.
//!
//! A [`Step`] is one reaction of one object, parsed once from the flat integers the Python
//! encoder writes. It reads and stores bytes through the interpreter's memory access, so a
//! device read or a store the interpreter refuses bails the frame exactly as an ordinary
//! statement would. `pysidtracker.playspec.objects` is the reference these steps follow.

use std::cell::Cell;

use crate::vm::{read, store, Bail, Env};

const JUMPS: usize = 256;

/// The byte addresses of a cell: low, and high for a word.
#[derive(Clone, Copy, Debug)]
struct Loc {
    lo: i64,
    hi: Option<i64>,
}

/// `(column, mask, shl, shr)`: a row byte through a mask and a shift, or the constant `mask`.
#[derive(Clone, Copy, Debug)]
struct Xf {
    col: i64,
    mask: i64,
    shl: i64,
    shr: i64,
}

/// `value[.part] = xf`: part 0 is the whole cell, 1 its low byte, 2 its high byte.
#[derive(Clone, Copy, Debug)]
struct Assign {
    index: usize,
    part: i64,
    xf: Xf,
    /// `(column, mask, value, equal)`: the row byte test the assignment waits for.
    guard: Option<(usize, i64, i64, bool)>,
}

/// `(column, mask, value, equal)`: a test of a row byte.
type Test = (usize, i64, i64, bool);

#[derive(Clone, Debug)]
enum Action {
    Stop,
    Jump {
        target: Xf,
        hold: (i64, i64),
        via: Option<i64>,
    },
    Sweep {
        into: usize,
        count: Xf,
        step: Xf,
        frac: u32,
    },
    Delay(Xf),
    Set(Vec<Assign>, i64),
    End(i64),
    Call(Xf),
    Return,
    Run(i64, Vec<Assign>),
}

#[derive(Clone, Debug)]
struct Row {
    lo: i64,
    hi: i64,
    tests: Vec<Test>,
    action: Action,
    then: Vec<Assign>,
}

#[derive(Clone, Debug)]
enum Kind {
    Counter {
        cell: Loc,
        action: i64,
        delta: i64,
    },
    Owner {
        cell: Loc,
        release: bool,
    },
    Accum {
        cell: Loc,
        mode: i64,
        dir: Option<Loc>,
    },
    Cursor(Box<Cursor>),
}

#[derive(Clone, Debug)]
struct Cursor {
    adv: i64,
    width: i64,
    columns: usize,
    reaction: i64,
    pos: Loc,
    cnt: Option<Loc>,
    run: Option<Loc>,
    ret: Option<Loc>,
    values: Vec<Loc>,
    rows: Vec<Row>,
}

/// One reaction of an object.
#[derive(Clone, Debug)]
pub struct Step {
    /// The variable the wrap flag goes to.
    pub out: Option<usize>,
    /// The operands it pops.
    pub operands: usize,
    kind: Kind,
}

struct Reader<'a> {
    data: &'a [i64],
    at: usize,
}

impl Reader<'_> {
    fn next(&mut self) -> Result<i64, String> {
        let v = *self
            .data
            .get(self.at)
            .ok_or("an object description ends early")?;
        self.at += 1;
        Ok(v)
    }

    fn count(&mut self) -> Result<usize, String> {
        usize::try_from(self.next()?).map_err(|_| "a negative count".to_string())
    }

    fn loc(&mut self) -> Result<Loc, String> {
        let (lo, hi) = (self.next()?, self.next()?);
        if !(0..=0xFFFF).contains(&lo) || hi > 0xFFFF {
            return Err("an address outside the 64 KiB".into());
        }
        Ok(Loc {
            lo,
            hi: (hi >= 0).then_some(hi),
        })
    }

    fn opt_loc(&mut self) -> Result<Option<Loc>, String> {
        if self.data.get(self.at).is_some_and(|&lo| lo < 0) {
            self.at += 2;
            return Ok(None);
        }
        self.loc().map(Some)
    }

    fn xf(&mut self) -> Result<Xf, String> {
        let xf = Xf {
            col: self.next()?,
            mask: self.next()?,
            shl: self.next()?,
            shr: self.next()?,
        };
        if !(0..64).contains(&xf.shl) || !(0..64).contains(&xf.shr) {
            return Err("a shift outside 0..64".into());
        }
        Ok(xf)
    }

    fn assigns(&mut self) -> Result<Vec<Assign>, String> {
        let n = self.count()?;
        (0..n)
            .map(|_| {
                let (index, part, xf) = (self.count()?, self.next()?, self.xf()?);
                let (col, mask, value, equal) =
                    (self.next()?, self.next()?, self.next()?, self.next()? != 0);
                Ok(Assign {
                    index,
                    part,
                    xf,
                    guard: usize::try_from(col).ok().map(|c| (c, mask, value, equal)),
                })
            })
            .collect()
    }
}

impl Step {
    /// Parses a description: `kind, out (-1: none), operands, ...` as the encoder writes it.
    pub fn parse(data: &[i64], vars: usize) -> Result<Step, String> {
        let mut r = Reader { data, at: 0 };
        let kind = r.next()?;
        let out = r.next()?;
        let operands = r.count()?;
        if out >= vars as i64 {
            return Err("an object's flag variable is out of range".into());
        }
        let kind = match kind {
            0 => Kind::Counter {
                cell: r.loc()?,
                action: r.next()?,
                delta: r.next()?,
            },
            1 => Kind::Owner {
                cell: r.loc()?,
                release: r.next()? != 0,
            },
            3 => {
                let (cell, mode, dir) = (r.loc()?, r.next()?, r.opt_loc()?);
                if !(0..4).contains(&mode) || (mode >= 2 && dir.is_none()) {
                    return Err("an accumulator mode needs a direction cell".into());
                }
                Kind::Accum { cell, mode, dir }
            }
            2 => Self::cursor(&mut r, operands)?,
            _ => return Err("no such object kind".into()),
        };
        if r.at != data.len() {
            return Err("an object description is too long".into());
        }
        Ok(Step {
            out: (out >= 0).then_some(out as usize),
            operands,
            kind,
        })
    }

    fn cursor(r: &mut Reader, operands: usize) -> Result<Kind, String> {
        let (adv, width, columns, reaction) = (r.next()?, r.next()?, r.count()?, r.next()?);
        let pos = r.loc()?;
        let (cnt, run, ret) = (r.opt_loc()?, r.opt_loc()?, r.opt_loc()?);
        let values = (0..r.count()?)
            .map(|_| r.loc())
            .collect::<Result<Vec<_>, _>>()?;
        let rows = (0..r.count()?)
            .map(|_| Self::row(r))
            .collect::<Result<Vec<_>, String>>()?;
        let bad = |a: &Assign| {
            a.index > values.len()
                || (a.index == values.len() && cnt.is_none())
                || a.guard.is_some_and(|g| g.0 >= columns)
                || (a.index < values.len() && a.part == 2 && values[a.index].hi.is_none())
        };
        let broken = rows.iter().any(|row| {
            row.then.iter().any(bad)
                || row.tests.iter().any(|t| t.0 >= columns)
                || match &row.action {
                    Action::Set(sets, take) => sets.iter().any(bad) || *take < 1,
                    Action::Run(_, sets) => run.is_none() || sets.iter().any(bad),
                    Action::Call(_) | Action::Return => ret.is_none(),
                    Action::Sweep { into, frac, .. } => {
                        cnt.is_none()
                            || *into >= values.len()
                            || (*frac > 0 && values[*into].hi.is_none())
                            || *frac > 24
                    }
                    Action::Delay(_) => cnt.is_none(),
                    _ => false,
                }
        });
        if broken
            || operands != columns + 1
            || columns == 0
            || columns > 16
            || !(0..3).contains(&adv)
            || !(0..3).contains(&reaction)
            || ret.is_some_and(|l| l.hi.is_some() != pos.hi.is_some())
        {
            return Err("a cursor refers to a value or operand it lacks".into());
        }
        Ok(Kind::Cursor(Box::new(Cursor {
            adv,
            width,
            columns,
            reaction,
            pos,
            cnt,
            run,
            ret,
            values,
            rows,
        })))
    }

    fn row(r: &mut Reader) -> Result<Row, String> {
        let (lo, hi) = (r.next()?, r.next()?);
        let tests = (0..r.count()?)
            .map(|_| {
                let (col, mask, value, equal) = (r.count()?, r.next()?, r.next()?, r.next()? != 0);
                Ok((col, mask, value, equal))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let (kind, take, into, frac, base) =
            (r.next()?, r.next()?, r.count()?, r.next()?, r.next()?);
        let (a, b) = (r.xf()?, r.xf()?);
        let sets = r.assigns()?;
        let then = r.assigns()?;
        let (hold, via) = ((r.next()?, r.next()?), r.next()?);
        let action = match kind {
            0 => Action::Stop,
            1 => Action::Jump {
                target: a,
                hold,
                via: (via >= 0).then_some(via),
            },
            2 => Action::Sweep {
                into,
                count: a,
                step: b,
                frac: u32::try_from(frac).map_err(|_| "a negative fraction")?,
            },
            3 => Action::Delay(a),
            4 => Action::Set(sets, take),
            5 => Action::End(take),
            6 => Action::Call(a),
            7 => Action::Return,
            8 => Action::Run(base, sets),
            _ => return Err("no such cursor row".to_string()),
        };
        Ok(Row {
            lo,
            hi,
            tests,
            action,
            then,
        })
    }

    /// Runs the reaction on `ops` (as many as [`Step::operands`]); the wrap flag.
    #[inline]
    pub fn run(
        &self,
        env: &mut Env,
        ram: &[Cell<u8>],
        slow: u16,
        ops: &[i64],
    ) -> Result<i64, Bail> {
        let mut mem = Mem { env, ram, slow };
        match &self.kind {
            Kind::Counter {
                cell,
                action,
                delta,
            } => counter(&mut mem, *cell, *action, *delta, ops),
            Kind::Owner { cell, release } => owner(&mut mem, *cell, *release, ops),
            Kind::Accum { cell, mode, dir } => accum(&mut mem, *cell, *mode, *dir, ops),
            Kind::Cursor(c) => cursor(&mut mem, c, ops),
        }
    }
}

struct Mem<'a, 'b> {
    env: &'a mut Env<'b>,
    ram: &'a [Cell<u8>],
    slow: u16,
}

impl Mem<'_, '_> {
    fn get(&mut self, addr: i64) -> Result<i64, Bail> {
        read(self.env, self.ram, self.slow, addr)
    }

    fn put(&mut self, addr: i64, value: i64) -> Result<(), Bail> {
        store(self.env, addr, value & 0xFF)
    }

    fn load(&mut self, loc: Loc) -> Result<i64, Bail> {
        let lo = self.get(loc.lo)?;
        Ok(match loc.hi {
            Some(hi) => lo | self.get(hi)? << 8,
            None => lo,
        })
    }

    fn save(&mut self, loc: Loc, value: i64) -> Result<(), Bail> {
        self.put(loc.lo, value)?;
        match loc.hi {
            Some(hi) => self.put(hi, value >> 8),
            None => Ok(()),
        }
    }
}

fn counter(m: &mut Mem, cell: Loc, action: i64, delta: i64, ops: &[i64]) -> Result<i64, Bail> {
    let (limit, reload) = (ops[0], ops[1]);
    let value = m.load(cell)?;
    if action == 0 {
        m.save(cell, reload)?;
        return Ok(0);
    }
    let up = action == 1 || action == 3;
    let after = if up { value + delta } else { value - delta };
    let wrap = match action {
        1 => value + delta > limit,
        2 => value < delta,
        _ => limit != 0,
    };
    m.save(cell, if wrap { reload } else { after })?;
    Ok(wrap as i64)
}

fn owner(m: &mut Mem, cell: Loc, release: bool, ops: &[i64]) -> Result<i64, Bail> {
    let (ident, none, cond) = (ops[0], ops[1], ops[2]);
    if cond != 0 && (!release || m.load(cell)? == ident) {
        m.save(cell, if release { none } else { ident })?;
    }
    Ok(0)
}

fn accum(m: &mut Mem, cell: Loc, mode: i64, dir: Option<Loc>, ops: &[i64]) -> Result<i64, Bail> {
    let (amount, target) = (ops[0], ops[1]);
    let value = m.load(cell)?;
    let top = if cell.hi.is_some() { 0xFFFF } else { 0xFF };
    let down = match dir {
        Some(d) if mode >= 2 => m.get(d.lo)?,
        _ => 0,
    };
    match mode {
        3 => {
            m.put(dir.map_or(0, |d| d.lo), down ^ 1)?;
            Ok(0)
        }
        1 => {
            if value == target {
                return Ok(1);
            }
            let moved = if value < target {
                (value + amount).min(target)
            } else {
                (value - amount).max(target)
            };
            m.save(cell, moved)?;
            Ok((moved == target) as i64)
        }
        _ => {
            let after = if down & 1 != 0 {
                value - amount
            } else {
                value + amount
            };
            m.save(cell, after & top)?;
            Ok((after < 0 || after > top) as i64)
        }
    }
}

fn xf(x: Xf, cols: &[i64]) -> i64 {
    if x.col < 0 {
        x.mask
    } else {
        (((cols[x.col as usize] & x.mask) << x.shl) >> x.shr) & 0xFF
    }
}

fn passes(t: &Test, cols: &[i64]) -> bool {
    ((cols[t.0] & t.1) == t.2) == t.3
}

fn assign(
    m: &mut Mem,
    values: &[Loc],
    sets: &[Assign],
    cols: &[i64],
    count: &mut i64,
) -> Result<(), Bail> {
    for a in sets {
        if a.guard
            .is_some_and(|(col, mask, value, equal)| ((cols[col] & mask) == value) != equal)
        {
            continue;
        }
        let v = xf(a.xf, cols);
        if a.index == values.len() {
            *count = v & 0xFF;
            continue;
        }
        let loc = values[a.index];
        match a.part {
            0 => m.save(loc, v)?,
            1 => m.put(loc.lo, v)?,
            _ => m.put(loc.hi.ok_or(Bail::Range)?, v)?,
        }
    }
    Ok(())
}

fn sweep(m: &mut Mem, loc: Loc, frac: u32, delta: i64) -> Result<i64, Bail> {
    if frac == 0 {
        let carry = (m.get(loc.lo)? + (delta & 0xFF) > 0xFF) as i64;
        let v = m.load(loc)?;
        m.save(loc, v + delta)?;
        return Ok(carry);
    }
    let hi = loc.hi.ok_or(Bail::Range)?;
    let low = (1i64 << frac) - 1;
    let total = ((m.get(hi)? << frac) | (m.get(loc.lo)? & low)) + delta;
    let total = total & ((1i64 << (8 + frac)) - 1);
    m.put(hi, total >> frac)?;
    m.put(loc.lo, total & low)?;
    Ok(0)
}

fn cursor(m: &mut Mem, c: &Cursor, ops: &[i64]) -> Result<i64, Bail> {
    let room = if c.pos.hi.is_some() { 0xFFFF } else { 0xFF };
    let mut left = 0;
    let mut cols = [0i64; 16];
    match c.reaction {
        1 => {
            m.save(c.pos, (ops[c.columns] - (c.adv == 1) as i64) & room)?;
            for cell in [c.cnt, c.run].into_iter().flatten() {
                m.put(cell.lo, 0)?;
            }
            return Ok(0);
        }
        2 => {
            let held = match c.run {
                Some(run) => m.get(run.lo)? != 0,
                None => false,
            };
            if !held {
                let at = m.load(c.pos)?;
                m.save(c.pos, (at + 1) & room)?;
            }
            return Ok(0);
        }
        _ => {}
    }
    let mut at = m.load(c.pos)?;
    if let Some(cnt) = c.cnt {
        left = m.get(cnt.lo)?;
    }
    if let Some(run) = c.run {
        let pending = m.get(run.lo)?;
        if pending != 0 {
            m.put(run.lo, pending - 1)?;
            let empty = c.rows.iter().find_map(|r| match &r.action {
                Action::Run(_, sets) => Some(sets),
                _ => None,
            });
            if let Some(sets) = empty {
                assign(m, &c.values, sets, &cols[..c.columns], &mut left)?;
            }
            return Ok(0);
        }
    }
    at = (at + (c.adv == 1) as i64) & room;
    let (mut carry, mut via, mut jumped, mut used) = (0, None, false, 0);
    let mut done = false;
    for _ in 0..JUMPS {
        for (j, col) in cols.iter_mut().enumerate().take(c.columns) {
            *col = m.get((ops[j] + at) & 0xFFFF)?;
        }
        let first = via.unwrap_or(cols[0]);
        let row = c
            .rows
            .iter()
            .find(|r| r.lo <= first && first <= r.hi && r.tests.iter().all(|t| passes(t, &cols)));
        let Some(row) = row else {
            carry = 1;
            done = true;
            break;
        };
        let here = &cols[..c.columns];
        match &row.action {
            Action::Stop => {
                carry = 1;
                done = true;
                break;
            }
            Action::Jump { .. } | Action::Call(_) | Action::Return => {
                let target = match &row.action {
                    Action::Jump { target, .. } => xf(*target, here),
                    Action::Call(target) => xf(*target, here),
                    _ => match c.ret {
                        Some(ret) => m.load(ret)?,
                        None => return Err(Bail::Range),
                    },
                };
                if let (Action::Call(_), Some(ret)) = (&row.action, c.ret) {
                    m.save(ret, (at + c.width) & room)?;
                }
                carry = (target >= at) as i64;
                jumped = true;
                done = true;
                if matches!(row.action, Action::Jump { .. }) && target == at {
                    break;
                }
                at = target;
                via = None;
                if let Action::Jump {
                    hold, via: next, ..
                } = &row.action
                {
                    via = *next;
                    if hold.0 <= hold.1 {
                        let c0 = m.get((ops[0] + at) & 0xFFFF)?;
                        if hold.0 <= c0 && c0 <= hold.1 {
                            left = 0;
                            break;
                        }
                    }
                }
                done = false;
                continue;
            }
            Action::Sweep {
                into,
                count,
                step,
                frac,
            } => {
                if left != xf(*count, here) {
                    left = (left + 1) & 0xFF;
                    let d = xf(*step, here);
                    carry = sweep(m, c.values[*into], *frac, d - 256 * (d >> 7))?;
                    done = true;
                    break;
                }
                left = 0;
                carry = 1;
                used = c.width;
            }
            Action::Delay(limit) => {
                if left != xf(*limit, here) {
                    left = (left + 1) & 0xFF;
                    carry = 0;
                    done = true;
                    break;
                }
                left = 0;
                carry = 1;
                used = c.width;
            }
            Action::Set(sets, take) => {
                assign(m, &c.values, sets, here, &mut left)?;
                carry = if jumped { carry } else { 0 };
                used = *take;
            }
            Action::End(take) => {
                carry = 1;
                used = *take;
            }
            Action::Run(base, sets) => {
                if let Some(run) = c.run {
                    m.put(run.lo, (cols[0] - base) & 0xFF)?;
                }
                assign(m, &c.values, sets, &[0i64; 16][..c.columns], &mut left)?;
                carry = 0;
                used = 1;
            }
        }
        if !matches!(row.action, Action::Run(..)) {
            assign(m, &c.values, &row.then, here, &mut left)?;
        }
        done = true;
        break;
    }
    if !done {
        return Err(Bail::Spec);
    }
    m.save(c.pos, (at + used - (c.adv > 0) as i64) & room)?;
    if let Some(cnt) = c.cnt {
        m.put(cnt.lo, left)?;
    }
    Ok(carry)
}
