//! Event-driven objects: counters, owners, walkers and cursors, stepped natively.
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
    Set(Vec<Assign>),
}

#[derive(Clone, Debug)]
struct Row {
    lo: i64,
    hi: i64,
    action: Action,
    then: Vec<Assign>,
}

#[derive(Clone, Debug)]
struct Store {
    out: Loc,
    keep: i64,
    shl: i64,
    shr: i64,
    mask: i64,
    value: i64,
    equal: bool,
}

#[derive(Clone, Debug)]
struct Field {
    mask: i64,
    value: i64,
    equal: bool,
    stores: Vec<Store>,
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
    Walker {
        width: i64,
        columns: usize,
        pos: Loc,
        cnt: i64,
        values: Vec<Loc>,
        rows: Vec<Row>,
        reset: bool,
    },
    Cursor {
        pos: i64,
        run: Option<(i64, i64, i64, i64)>,
        fields: Vec<Field>,
        last: bool,
        mode: i64,
    },
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
            2 => Self::walker(&mut r, operands)?,
            3 => Self::cursor(&mut r)?,
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

    fn walker(r: &mut Reader, operands: usize) -> Result<Kind, String> {
        let width = r.next()?;
        let columns = r.count()?;
        let pos = r.loc()?;
        let cnt = r.loc()?.lo;
        let values = (0..r.count()?)
            .map(|_| r.loc())
            .collect::<Result<Vec<_>, _>>()?;
        let rows = (0..r.count()?)
            .map(|_| {
                let (lo, hi, kind, into, frac) =
                    (r.next()?, r.next()?, r.next()?, r.count()?, r.next()?);
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
                    4 => Action::Set(sets),
                    _ => return Err("no such walker row".to_string()),
                };
                Ok(Row {
                    lo,
                    hi,
                    action,
                    then,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let reset = r.next()? != 0;
        let bad = |a: &Assign| {
            a.index > values.len()
                || a.guard.is_some_and(|g| g.0 >= columns)
                || (a.index < values.len() && a.part == 2 && values[a.index].hi.is_none())
        };
        let broken = rows.iter().any(|row| {
            row.then.iter().any(bad)
                || match &row.action {
                    Action::Set(sets) => sets.iter().any(bad),
                    Action::Sweep { into, frac, .. } => {
                        *into >= values.len()
                            || (*frac > 0 && values[*into].hi.is_none())
                            || *frac > 24
                    }
                    _ => false,
                }
        });
        if broken || operands != columns + 1 || columns == 0 {
            return Err("a walker refers to a value or operand it lacks".into());
        }
        Ok(Kind::Walker {
            width,
            columns,
            pos,
            cnt,
            values,
            rows,
            reset,
        })
    }

    fn cursor(r: &mut Reader) -> Result<Kind, String> {
        let pos = r.loc()?.lo;
        let run = if r.next()? >= 0 {
            Some((r.next()?, r.next()?, r.next()?, r.loc()?.lo))
        } else {
            None
        };
        let last = r.next()? != 0;
        let fields = (0..r.count()?)
            .map(|_| {
                let (mask, value, equal) = (r.next()?, r.next()?, r.next()? != 0);
                let stores = (0..r.count()?)
                    .map(|_| {
                        Ok(Store {
                            out: r.loc()?,
                            keep: r.next()?,
                            shl: r.next()?,
                            shr: r.next()?,
                            mask: r.next()?,
                            value: r.next()?,
                            equal: r.next()? != 0,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Ok(Field {
                    mask,
                    value,
                    equal,
                    stores,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mode = r.next()?;
        if fields.is_empty()
            || !(0..3).contains(&mode)
            || fields.iter().any(|f| {
                f.stores
                    .iter()
                    .any(|s| !(0..64).contains(&s.shl) || !(0..64).contains(&s.shr))
            })
        {
            return Err("a cursor needs fields with shifts below 64".into());
        }
        Ok(Kind::Cursor {
            pos,
            run,
            fields,
            last,
            mode,
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
            Kind::Walker { .. } => walker(&mut mem, &self.kind, ops),
            Kind::Cursor { .. } => cursor(&mut mem, &self.kind, ops),
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

fn xf(x: Xf, cols: &[i64]) -> i64 {
    if x.col < 0 {
        x.mask
    } else {
        (((cols[x.col as usize] & x.mask) << x.shl) >> x.shr) & 0xFF
    }
}

fn assign(
    m: &mut Mem,
    values: &[Loc],
    sets: &[Assign],
    cols: &[i64],
    count: &mut i64,
) -> Result<(), Bail> {
    for a in sets {
        if let Some((col, mask, value, equal)) = a.guard {
            if ((cols[col] & mask) == value) != equal {
                continue;
            }
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

fn walker(m: &mut Mem, kind: &Kind, ops: &[i64]) -> Result<i64, Bail> {
    let Kind::Walker {
        width,
        columns,
        pos,
        cnt,
        values,
        rows,
        reset,
    } = kind
    else {
        unreachable!()
    };
    let room = if pos.hi.is_some() { 0xFFFF } else { 0xFF };
    if *reset {
        m.save(*pos, ops[*columns] & room)?;
        m.put(*cnt, 0)?;
        return Ok(0);
    }
    let (mut at, mut count) = (m.load(*pos)?, m.get(*cnt)?);
    let (mut carry, mut via, mut jumped) = (0, None, false);
    let mut cols = [0i64; 16];
    let mut done = false;
    for _ in 0..JUMPS {
        for (j, col) in cols.iter_mut().enumerate().take(*columns) {
            *col = m.get((ops[j] + at) & 0xFFFF)?;
        }
        let first = via.unwrap_or(cols[0]);
        let row = rows.iter().find(|r| r.lo <= first && first <= r.hi);
        let Some(row) = row else {
            carry = 1;
            done = true;
            break;
        };
        match &row.action {
            Action::Stop => {
                carry = 1;
                done = true;
                break;
            }
            Action::Jump {
                target,
                hold,
                via: next,
            } => {
                let target = xf(*target, &cols);
                carry = (target >= at) as i64;
                done = true;
                if target == at {
                    break;
                }
                jumped = true;
                at = target;
                via = *next;
                if hold.0 <= hold.1 {
                    let c0 = m.get((ops[0] + at) & 0xFFFF)?;
                    if hold.0 <= c0 && c0 <= hold.1 {
                        count = 0;
                        break;
                    }
                }
                done = false;
                continue;
            }
            Action::Sweep {
                into,
                count: limit,
                step,
                frac,
            } => {
                if count != xf(*limit, &cols) {
                    count = (count + 1) & 0xFF;
                    let d = xf(*step, &cols);
                    carry = sweep(m, values[*into], *frac, d - 256 * (d >> 7))?;
                    done = true;
                    break;
                }
                count = 0;
                carry = 1;
            }
            Action::Delay(limit) => {
                if count != xf(*limit, &cols) {
                    count = (count + 1) & 0xFF;
                    carry = 0;
                    done = true;
                    break;
                }
                count = 0;
                carry = 1;
            }
            Action::Set(sets) => {
                assign(m, values, sets, &cols, &mut count)?;
                carry = if jumped { carry } else { 0 };
            }
        }
        at = (at + width) & room;
        assign(m, values, &row.then, &cols, &mut count)?;
        done = true;
        break;
    }
    if !done {
        return Err(Bail::Spec);
    }
    m.save(*pos, at)?;
    m.put(*cnt, count)?;
    Ok(carry)
}

fn cursor(m: &mut Mem, kind: &Kind, ops: &[i64]) -> Result<i64, Bail> {
    let Kind::Cursor {
        pos,
        run,
        fields,
        last,
        mode,
    } = kind
    else {
        unreachable!()
    };
    if *mode == 1 {
        m.put(*pos, ops[1])?;
        if let Some((_, _, _, pending)) = run {
            m.put(*pending, 0)?;
        }
        return Ok(0);
    }
    if *mode == 2 {
        let held = match run {
            Some((_, _, _, pending)) => m.get(*pending)? != 0,
            None => false,
        };
        if !held {
            let at = m.get(*pos)?;
            m.put(*pos, at + 1)?;
        }
        return Ok(0);
    }
    let at = m.get(*pos)?;
    let (mut used, mut first, mut empty) = (0, 0, false);
    let left = match run {
        Some((_, _, _, pending)) => m.get(*pending)?,
        None => 0,
    };
    if left != 0 {
        m.put(run.unwrap().3, left - 1)?;
        empty = true;
    } else {
        first = m.get((ops[0] + at) & 0xFFFF)?;
        used = 1;
        if let Some((lo, hi, base, pending)) = run {
            if *lo <= first && first <= *hi {
                m.put(*pending, first - base)?;
                first = 0;
                empty = true;
            }
        }
    }
    let mut row = [0i64; 16];
    row[0] = first;
    let mut n = 1;
    while !empty && n < fields.len() {
        let f = &fields[n];
        if ((row[n - 1] & f.mask) == f.value) != f.equal {
            break;
        }
        row[n] = m.get((ops[0] + at + used) & 0xFFFF)?;
        used += 1;
        n += 1;
    }
    for (byte, f) in row.iter().zip(fields).take(n) {
        for s in &f.stores {
            if ((byte & s.mask) == s.value) == s.equal {
                m.save(s.out, (((byte & s.keep) << s.shl) >> s.shr) & 0xFF)?;
            }
        }
    }
    let used = if *last { (used - 1).max(0) } else { used };
    m.put(*pos, at + used)?;
    Ok(0)
}
