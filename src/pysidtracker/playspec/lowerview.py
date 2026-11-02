"""Lowering of bit fields and tables: named views of cells and memory (a mixin of the lowerer)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from .ir import Cell, K, Rd, V
from .lowerops import mk, word, zext

SIZES = {"u8": 1, **{f"u{n}": 2 for n in range(9, 17)}}
REGISTERS = 25


class ViewLowering:
    """Bit fields of cells and ``table`` declarations of a :class:`~.lower.Lowerer`."""

    def bind_bits(self, name: str, typ: str, bits, scope):
        """``{field: (low, high)}`` of a cell's named bit ranges (None: no fields)."""
        if bits is None:
            return None
        top = 8 * SIZES.get(typ, 2)
        out = {}
        for field, (_tag, bline, lo, hi) in bits.items():
            low, high = (self.static(e, scope, bline) for e in (lo, hi))
            if low is None or high is None:
                continue
            if not 0 <= low <= high < top:
                self.error(
                    bline, f"cell {name}.{field}: bits {low}..{high} not in u{top}"
                )
            out[field] = (low, high)
        return out

    def bind_output(self, scope) -> None:
        """The ``output`` declaration: the register block cell, the write order and, for a
        staging copy, the SID base the frame ends by copying it to."""
        self.outorder, self.sid, self.stage = None, None, None
        if len(self.spec.outputs) > 1:
            self.error(self.spec.outputs[1][1], "a spec has one output declaration")
        for _tag, line, name, at, order, copy in self.spec.outputs[:1]:
            size = ("num", line, REGISTERS)
            decl = (
                "cell",
                line,
                name,
                "u8",
                size,
                ("at", line, at),
                frozenset({"out"}),
                None,
                None,
            )
            self.shared[name] = self.bind_cell(decl, "", scope)
            self.outorder = order.strip()
            base = self.shared[name].place[0]
            self.sid = base if copy is None else self.static(copy, scope, line)
            self.stage = None if copy is None else base

    def flush(self) -> list:
        """The statements copying the staging block to the SID, in register order."""
        if self.stage is None or self.sid is None:
            return []
        return [
            ("wr", K(self.sid + r, 16), Rd(K(self.stage + r, 16)))
            for r in range(REGISTERS)
        ]

    def bit_cell(self, name: str, scope, attr: str) -> Optional[Cell]:
        """The cell ``name`` if it declares the bit field ``attr``."""
        found = self.lookup(name, scope)
        if found is not None and found[0] == "cell" and attr in (found[1].bits or ()):
            return self.in_proc(found[1])
        return None

    def bit_span(self, cell: Cell, attr: str, line: int) -> Tuple[int, int]:
        """``(shift, mask)`` of the bit field ``attr`` of ``cell``."""
        if attr not in cell.bits:
            self.error(line, f"cell {cell.name} has no bit field {attr!r}")
            return 0, 0
        low, high = cell.bits[attr]
        return low, (1 << high - low + 1) - 1

    def bit_value(self, name: str, index, attr: str, scope, line: int):
        """``cell.field`` (``cell[i].field``): the bit field, right-aligned."""
        cell = self.bit_cell(name, scope, attr)
        low, mask = self.bit_span(cell, attr, line)
        w = 8 * SIZES.get(cell.typ, 2)
        value = self.read_cell(cell, index, line)
        return mk("and", w, mk("shr", w, value, K(low, w)), K(mask, w))

    def write_bits(self, name: str, index, attr: str, value, scope, line: int):
        """Statements storing ``value`` to the bit field ``cell.field``."""
        cell = self.bit_cell(name, scope, attr)
        low, mask = self.bit_span(cell, attr, line)
        w = 8 * SIZES.get(cell.typ, 2)
        keep = K(~(mask << low) & (1 << w) - 1, w)
        put = mk("and", w, self.coerce(value, w, line), K(mask, w))
        put = mk("shl", w, put, K(low, w))
        merged = mk("or", w, mk("and", w, self.read_cell(cell, index, line), keep), put)
        return self.write_cell(cell, index, merged, line)

    def bind_tables(self, scope) -> None:
        """Place the fields of every ``table`` (a field follows the previous unless `@`)."""
        self.table_fields: Dict[str, Dict[str, Tuple[int, str]]] = {}
        for (
            _tag,
            line,
            name,
            base,
            via,
            stride,
            _offset,
            fields,
            within,
        ) in self.tables.values():
            size = self.static(stride, scope, line)
            placed: Dict[str, Tuple[int, str]] = {}
            taken: Dict[int, str] = {}
            at = 0
            for _t, fline, fname, typ, where in fields:
                at = at if where is None else self.static(where, scope, fline) or 0
                if typ not in SIZES:
                    self.error(fline, f"table field {fname}: u8 or u16 only")
                    continue
                for byte in range(at, at + SIZES[typ]):
                    if taken.setdefault(byte, fname) != fname:
                        self.error(fline, f"table field {fname} overlaps {taken[byte]}")
                if size is not None and at + SIZES[typ] > size:
                    self.error(
                        fline, f"table field {fname} ends past the stride {size}"
                    )
                placed[fname] = (at, typ)
                at += SIZES[typ]
            self.table_fields[name] = placed
            if name in self.shared:
                self.error(line, f"table {name} is already a cell")
            if [base, via, within].count(None) != 2:
                self.error(line, f"table {name}: `@ base`, `via pointer` or `in array`")

    def table_ref(self, name: str, picks, field: str, scope, line: int):
        """``(address, type)`` of ``name[picks...].field``: the row index last, a pointer
        table's sub-table index first."""
        _tag, _l, _name, base, via, stride, offset, _fields, _within = self.tables[name]
        where = self.table_fields[name].get(field)
        if where is None:
            self.error(line, f"table {name} has no field {field!r}")
            return K(None, 16), "u8"
        start = K(0, 16)
        wanted = 1
        if base is not None:
            start = self.coerce(self.expr(base, scope), 16, line)
        elif via is not None:
            found = self.lookup(via, scope)
            cell = self.in_proc(found[1]) if found and found[0] == "cell" else None
            if (
                found
                and found[0] == "var"
                and found[1] in self.state_widths
                and found[2] == 16
            ):
                start = V(found[1], 16)
            elif cell is None or cell.typ not in ("u16", "ptr"):
                self.error(line, f"table {name}: {via} is not a u16 cell")
            else:
                wanted = 1 + (cell.length is not None)
                select = picks[0] if cell.length is not None else None
                start = (
                    self.read_cell(cell, select, line)
                    if len(picks) == wanted
                    else start
                )
        if len(picks) != wanted:
            self.error(line, f"table {name} takes {wanted} index(es)")
            return K(None, 16), where[1]
        if offset is not None:
            start = mk(
                "add", 16, start, self.coerce(self.expr(offset, scope), 16, line)
            )
        row = self.index_expr(picks[-1], line)
        step = self.static(stride, scope, line)
        span = mk("mul", 16, zext(row, 16), K(step, 16))
        return mk("add", 16, mk("add", 16, start, span), K(where[0], 16)), where[1]

    def table_value(self, name: str, picks, field: str, scope, line: int):
        """The value of ``name[picks...].field``."""
        addr, typ = self.table_ref(name, picks, field, scope, line)
        read = self.rd
        within = self.tables[name][8]
        if within is not None:
            arr = self.array_ref(within, scope, line, ro=True)
            if arr is None:
                return K(None, 8)

            def read(at):
                return self.array_read(arr, at, line)

        if typ == "u8":
            return read(addr)
        return word(read(addr), read(mk("add", 16, addr, K(1, 16))))

    def table_write(
        self, name: str, picks, field: str, value, scope, line: int
    ) -> list:
        """Statements storing ``value`` to ``name[picks...].field``."""
        if self.tables[name][8] is not None:
            self.error(line, f"write to table {name}, which reads a ro array")
            return []
        addr, typ = self.table_ref(name, picks, field, scope, line)
        if typ == "u8":
            return [("wr", addr, self.coerce(value, 8, line))]
        value = self.coerce(value, 16, line)
        pre = []
        if not isinstance(value, (K, V)):
            tmp = self.fresh("w16")
            pre, value = [("set", tmp, value)], V(tmp, 16)
        if not isinstance(addr, (K, V)):
            tmp = self.fresh("at16")
            pre, addr = pre + [("set", tmp, addr)], V(tmp, 16)
        return pre + [
            ("wr", addr, mk("lo", 8, value)),
            ("wr", mk("add", 16, addr, K(1, 16)), mk("hi", 8, value)),
        ]


__all__ = ["ViewLowering"]
