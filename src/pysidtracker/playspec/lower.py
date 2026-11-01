"""Elaborate a parsed spec against a layout binding: static checks plus lowering.

With ``binding`` None every ``L.*`` value is unknown and both arms of a static
choice are checked; with a binding the frame is specialised per instance, procs
are inlined (returns become guard flags) and static choices fold.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..notes import PARTS
from .ir import Cell, K, Obj, Op, Program, Rd, V, width
from .lowerobj import ObjectLowering
from .lowerops import add16, mk, word, zext
from .syntax import OBJECTS, Spec, SpecError

WIDTHS = {"u8": 8, "u16": 16, "bool": 1, "ptr": 16}
ARITH = {"+": "add", "-": "sub", "*": "mul", "&": "and", "|": "or", "^": "xor"}
SHIFT = {"<<": "shl", ">>": "shr"}
COMPARE = {"==": "eq", "!=": "ne", "<": "ult", "<=": "ule", ">": "ugt", ">=": "uge"}
COMPOUND = {
    "+=": "+",
    "-=": "-",
    "&=": "&",
    "|=": "|",
    "^=": "^",
    "<<=": "<<",
    ">>=": ">>",
}
LKINDS = {"addr", "byte", "flag", "list"}
CPU_INPUTS = {"a": 8, "x": 8, "y": 8, **dict.fromkeys("czvndi", 1)}
UNKNOWN_LIST = ("?",)


class Scope:
    """Local names of one inlined body: ``name -> ("var", ir, w) | ("stream", base, pos)
    | ("static", K)``."""

    def __init__(self, names=None):
        self.names: Dict[str, tuple] = dict(names or {})

    def child(self) -> "Scope":
        """A nested scope seeing this one's names."""
        return Scope(self.names)


class Instance:
    """A machine instance: its index, private cells, signals and controls."""

    def __init__(self, ident: str, machine: Optional[tuple], x: Optional[K]):
        self.ident = ident
        self.machine = machine
        self.x = x
        self.cells: Dict[str, Cell] = {}
        self.signals: Dict[str, Tuple[str, int]] = {}
        self.controls: Dict[str, tuple] = {}
        self.procs: Dict[str, tuple] = {}
        self.objects: Dict[str, Obj] = {}


class Lowerer(ObjectLowering):
    """One elaboration of a :class:`Spec` (see :func:`elaborate`)."""

    # pylint: disable=too-many-public-methods,too-many-instance-attributes

    def __init__(
        self, spec: Spec, binding: Optional[dict], standard: Optional[int] = None
    ):
        self.spec = spec
        self.binding = binding
        self.standard = standard
        self.errors: Dict[Tuple[int, str], None] = {}
        self.uid = 0
        self.natives: list = []
        self.consts: Dict[str, K] = {}
        self.shared: Dict[str, Cell] = {}
        self.instances: Dict[str, Instance] = {}
        self.signals: List[Tuple[str, int, int]] = []
        self.controls: List[Tuple[str, int, int]] = []
        self.grammars: List[Tuple[str, int]] = []
        self.ran: set = set()
        self.writers: Dict[str, set] = {}
        self.order: List[str] = []
        self.procs: Dict[str, list] = {}
        self.lowered: Dict[tuple, tuple] = {}
        self.latched: Optional[list] = None
        self.inputs: Dict[str, int] = {}
        self.stack: List[str] = []
        self.proc = ""
        self.decls: Dict[int, tuple] = {}
        self.variants: Dict[Tuple[int, str], Cell] = {}
        self.place_of: Optional[Tuple[str, str]] = None
        self.inst: Instance = Instance("", None, None)
        self.ret: Optional[Tuple[str, Optional[str], int]] = None
        self.allow_old = False
        self.old = False
        self.global_inst = self.inst
        self.events: Dict[str, Optional[str]] = {}
        self.shared_objs: Dict[str, Obj] = {}
        self.asts: Dict[str, dict] = {}
        self.owner_exprs: Dict[str, tuple] = {}
        self.bind_events()
        for kind in set(spec.layout.values()) - LKINDS:
            self.error(0, f"unknown layout kind {kind!r}")
        for _tag, line, name, expr in spec.consts:
            value = self.expr(expr, Scope())
            if not isinstance(value, K):
                self.error(line, f"const {name} is not static")
                value = K(None, 0)
            self.consts[name] = value
        for decl in spec.cells:
            self.shared[decl[2]] = self.bind_cell(decl, "", Scope())
        for record in spec.records:
            self.bind_record(record, self.shared, Scope())
        for item in spec.objects:
            self.declare_reacting(item, self.shared_objs, Scope())
        for name, machine in spec.machines.items():
            if machine[3] is None:
                self.instance(name, None)

    # -- diagnostics ---------------------------------------------------------
    def error(self, line: int, message: str) -> None:
        """Record a diagnostic."""
        self.errors[(line, message)] = None

    def fresh(self, stem: str) -> str:
        """A unique IR variable name."""
        self.uid += 1
        return f"{stem}_{self.uid}"

    def raise_errors(self) -> None:
        """Raise :class:`SpecError` listing every diagnostic, if any."""
        if self.errors:
            lines = [f"line {line}: {msg}" for line, msg in sorted(self.errors)]
            raise SpecError("\n".join(lines))

    # -- statics -------------------------------------------------------------
    def static(self, expr, scope: Scope, line: int) -> Optional[int]:
        """The value of a static integer expression (None while unknown)."""
        value = self.expr(expr, scope)
        if not isinstance(value, K):
            self.error(line, "expression must be static")
            return None
        return value.v

    def layout(self, name: str, line: int):
        """``L.name``: a static K, or a tuple for a ``list`` field."""
        kind = self.spec.layout.get(name)
        if kind is None:
            self.error(line, f"L.{name} is not declared in the layout block")
            return K(None, 0)
        if self.binding is None:
            return (
                UNKNOWN_LIST if kind == "list" else K(None, 1 if kind == "flag" else 0)
            )
        key = name
        if self.place_of is not None and name == self.place_of[0]:
            key = self.place_of[1]
        if key not in self.binding:
            self.error(line, f"binding has no layout field {name!r}")
            return UNKNOWN_LIST if kind == "list" else K(None, 0)
        value = self.binding[key]
        if kind == "list":
            return tuple(int(v) for v in value)
        if kind == "flag":
            return K(int(bool(value)), 1)
        return K(int(value), 0)

    # -- cells ---------------------------------------------------------------
    def bind_cell(self, decl, owner: str, scope: Scope) -> Cell:
        """Bind a ``cell`` declaration's addresses."""
        _tag, line, name, typ, length, place, attrs, notes = decl
        if typ == "bool":
            self.error(line, f"cell {name}: bool is not a memory type")
        size = None if length is None else self.static(length, scope, line)
        if length is not None and size is None and self.binding is not None:
            size = -1
        addrs = tuple(self.static(p, scope, line) for p in place[2:])
        if len(addrs) == 2 and typ == "u8":
            self.error(line, f"cell {name}: a split place needs a ptr or u16 type")
        if typ == "ptr" and length is None:
            typ = "u16"
        bad = attrs & {"ro", "out"} == {"ro", "out"}
        if bad:
            self.error(line, f"cell {name} cannot be both ro and out")
        place16 = tuple(None if a is None else a & 0xFFFF for a in addrs)
        cell = Cell(
            name,
            owner,
            typ,
            size,
            place16,
            frozenset(attrs),
            line,
            self.bind_notes(name, typ, length, attrs, notes, scope, line),
        )
        if "per_proc" in attrs:
            self.decls[id(cell)] = (decl, owner, scope)
        return cell

    def in_proc(self, cell: Cell) -> Cell:
        """``cell`` as the current proc addresses it: a ``per_proc`` cell whose binding
        has ``<cell>@<proc>`` is rebound there (``L.<cell>`` read as that field)."""
        known = self.decls.get(id(cell))
        key = f"{cell.name}@{self.proc}"
        if known is None or self.binding is None or key not in self.binding:
            return cell
        if (id(cell), self.proc) not in self.variants:
            decl, owner, scope = known
            self.place_of = (cell.name, key)
            try:
                self.variants[(id(cell), self.proc)] = self.bind_cell(
                    decl, owner, scope
                )
            finally:
                self.place_of = None
        return self.variants[(id(cell), self.proc)]

    def bind_notes(self, name, typ, length, attrs, notes, scope: Scope, line: int):
        """``(part, offset)`` of a cell's ``notes(part, offset)`` (None: undeclared)."""
        if notes is None:
            return None
        part, expr = notes
        offset = self.static(expr, scope, line)
        bad = "ro" not in attrs or length is None or (part == "word") != (typ != "u8")
        if bad:
            self.error(
                line,
                f"cell {name}: notes needs a ro table, lo/hi of a u8[n] or word of a u16[n]/ptr[n]",
            )
        return None if offset is None else (part, offset)

    def instance(self, machine_name: str, index: Optional[int]) -> Instance:
        """The (lazily bound) instance of ``machine_name`` at static ``index``."""
        machine = self.spec.machines[machine_name]
        ident = (
            machine_name
            if machine[3] is None
            else f"{machine_name}[{'?' if index is None else index}]"
        )
        if ident in self.instances:
            return self.instances[ident]
        x = None if machine[3] is None else K(index, 0)
        inst = Instance(ident, machine, x)
        scope = Scope({machine[3]: ("static", x)} if machine[3] else {})
        self.instances[ident] = inst
        saved, self.inst = self.inst, inst
        for item in machine[6]:
            tag = item[0]
            if tag == "cell":
                inst.cells[item[2]] = self.bind_cell(item, ident, scope)
            elif tag == "signal":
                self.bind_signal(inst, item, scope)
            elif tag == "control":
                self.bind_control(inst, item, scope)
            elif tag == "proc":
                inst.procs[item[2]] = item
            elif tag == "record":
                self.bind_record(item, inst.cells, scope)
        for item in machine[6]:
            if item[0] in OBJECTS:
                self.declare_reacting(item, inst.objects, scope)
        self.inst = saved
        return inst

    def bind_signal(self, inst: Instance, item, scope: Scope) -> None:
        """Declare a frame-local signal with its per-frame default."""
        _tag, line, name, typ, default = item
        w = WIDTHS.get(typ, 8)
        value = self.coerce(self.expr(default, scope), w, line)
        if not isinstance(value, K):
            self.error(line, f"signal {name} default must be static")
            value = K(0, w)
        ir = self.fresh(f"sig_{name}")
        inst.signals[name] = (ir, w)
        self.signals.append((ir, w, value.v or 0))

    def bind_control(self, inst: Instance, item, scope: Scope) -> None:
        """Declare a control state set, memory-mapped or hidden."""
        _tag, line, name, cell, mask, states = item
        values = {}
        for k, (_s, sline, sname, sval) in enumerate(states):
            value = k if sval is None else self.static(sval, scope, sline)
            values[sname] = value
        if cell is not None:
            target = inst.cells.get(cell) or self.shared.get(cell)
            if target is None or target.length is not None or target.typ != "u8":
                self.error(line, f"control {name}: {cell} is not a scalar u8 cell")
            inst.controls[name] = ("cell", cell, self.static(mask, scope, line), values)
        else:
            ir = self.fresh(f"ctl_{name}")
            self.controls.append((ir, 8, next(iter(values.values()), 0) or 0))
            inst.controls[name] = ("hidden", ir, 0xFF, values)

    def cell_access(self, cell: Cell, index, line: int):
        """``(byte addresses, combine)`` of a cell read at optional ``index``."""
        if (cell.length is None) != (index is None):
            self.error(
                line,
                f"cell {cell.name} {'is not' if index is not None else 'needs'} an index",
            )
            return [K(None, 16)], None
        place = [K(p, 16) for p in cell.place]
        if cell.typ == "u8":
            return [place[0] if index is None else add16(place[0], index)], None
        if len(place) == 2:
            if index is None:
                return place, word
            return [add16(p, index) for p in place], word
        if index is None:
            return [place[0], mk("add", 16, place[0], K(1, 16))], word
        offset = mk("shl", 16, zext(index, 16), K(1, 16))
        lo = mk("add", 16, place[0], offset)
        return [lo, mk("add", 16, lo, K(1, 16))], word

    def read_cell(self, cell: Cell, index, line: int):
        """The value of ``cell`` (``[index]``)."""
        cell = self.in_proc(cell)
        self.check_latched(cell, line)
        index = self.index_expr(index, line)
        if cell.notes is not None and self.standard is not None and index is not None:
            part, offset = cell.notes
            return Op(
                "note",
                16 if part == "word" else 8,
                (K(self.standard, 0), K(PARTS[part], 0), K(offset, 0), index),
            )
        addrs, combine = self.cell_access(cell, index, line)
        volatile = "volatile" in cell.attrs
        reads = [self.rd(a, volatile) for a in addrs]
        return reads[0] if combine is None else combine(*reads)

    def check_latched(self, cell: Cell, line: int) -> None:
        """Report a read of a latched ``cell`` that another instance wrote this frame."""
        if "latched" not in cell.attrs:
            return
        if self.latched is not None:
            self.latched.append((cell, line))
        others = self.writers.get(self.cell_key(cell), set()) - {self.inst.ident}
        if others:
            self.error(
                line,
                f"causality: {self.inst.ident or 'schedule'} reads latched cell"
                f" {cell.name} after {', '.join(sorted(others))} wrote it this frame",
            )

    def rd(self, addr, volatile: bool = False):
        """A byte read (marked for frame-start memory inside ``old()``); a ``volatile``
        read is an input of the call, never folded (so not built by :func:`mk`)."""
        if volatile:
            return Op("vol", 8, (addr,))
        return Op("old", 8, (Rd(addr),)) if self.old else Rd(addr)

    def cell_key(self, cell: Cell) -> str:
        """A cell's frame-wide identity."""
        return f"{cell.owner}.{cell.name}"

    def write_cell(self, cell: Cell, index, value, line: int) -> list:
        """Statements storing ``value`` to ``cell`` (``[index]``)."""
        cell = self.in_proc(cell)
        if "ro" in cell.attrs:
            self.error(line, f"write to read-only cell {cell.name}")
        if "volatile" in cell.attrs:
            self.error(line, f"write to volatile cell {cell.name}")
        self.writers.setdefault(self.cell_key(cell), set()).add(self.inst.ident)
        addrs, combine = self.cell_access(cell, self.index_expr(index, line), line)
        if combine is None:
            return [("wr", addrs[0], self.coerce(value, 8, line))]
        value = self.coerce(value, 16, line)
        pre = []
        if not isinstance(value, (K, V)):
            tmp = self.fresh("w16")
            pre, value = [("set", tmp, value)], V(tmp, 16)
        return pre + [
            ("wr", addrs[0], mk("lo", 8, value)),
            ("wr", addrs[1], mk("hi", 8, value)),
        ]

    def index_expr(self, index, line: int):
        """An index operand as an 8- or 16-bit expression (None passes through)."""
        if index is None:
            return None
        if width(index) == 1:
            self.error(line, "a truth value cannot index")
        return K(index.v, 16) if isinstance(index, K) and index.w == 0 else index

    # -- names ---------------------------------------------------------------
    def lookup(self, name: str, scope: Scope):
        """``(kind, payload)`` of a bare name."""
        if name in scope.names:
            return scope.names[name]
        inst = self.inst
        if inst.machine is not None and inst.machine[3] == name:
            return ("static", inst.x)
        if name in inst.signals:
            return ("signal", inst, name)
        if name in inst.cells:
            return ("cell", inst.cells[name])
        if name in self.consts:
            return ("static", self.consts[name])
        if name in self.shared:
            return ("cell", self.shared[name])
        return None

    def signal_read(self, inst: Instance, name: str, line: int):
        """A signal of ``inst``; a foreign read must follow its writer in the schedule."""
        if inst is not self.inst and inst.ident not in self.ran:
            self.error(
                line,
                f"causality: {self.inst.ident or 'schedule'} reads {inst.ident}.{name}"
                f" before {inst.ident} runs this frame",
            )
        ir, w = inst.signals[name]
        return V(ir, w)

    def member(self, inst: Instance, attr: str, line: int):
        """``instance.attr``: a signal or private cell of another instance."""
        if attr in inst.signals:
            return self.signal_read(inst, attr, line)
        if attr in inst.cells:
            return self.read_cell(inst.cells[attr], None, line)
        self.error(line, f"{inst.ident} has no signal or cell {attr!r}")
        return K(None, 8)

    def machine_instance(self, name: str, index, line: int) -> Optional[Instance]:
        """The instance ``name`` / ``name[index]`` names, or None after a diagnostic."""
        machine = self.spec.machines.get(name)
        if machine is None:
            self.error(line, f"undefined machine {name!r}")
            return None
        if (machine[3] is None) != (index is None):
            self.error(
                line,
                f"machine {name} {'is not' if machine[3] is None else 'must be'} indexed",
            )
            return None
        if index is None:
            return self.instance(name, None)
        if not isinstance(index, K):
            self.error(line, f"{name}[...] needs a static index")
            return None
        if index.v is not None:
            lo = self.static(machine[4], Scope(), line)
            hi = self.static(machine[5], Scope(), line)
            if lo is not None and hi is not None and not lo <= index.v < hi:
                self.error(line, f"{name}[{index.v}] is outside {lo}..{hi}")
                return None
        return self.instance(name, index.v)

    # -- expressions ---------------------------------------------------------
    def coerce(self, expr, w: int, line: int):
        """``expr`` as width ``w``: widening is implicit; narrowing is a width error."""
        have = width(expr)
        if have == w:
            return expr
        if isinstance(expr, K) and have == 0:
            if w == 1:
                self.error(
                    line, "width error: an integer where a truth value is needed"
                )
                return K(expr.v, 1)
            if expr.v is not None and not -(1 << w) < expr.v < (1 << w):
                self.error(line, f"width error: {expr.v} does not fit u{w}")
            return K(None if expr.v is None else expr.v & ((1 << w) - 1), w)
        if have == 1 or w == 1:
            self.error(
                line,
                f"width error: {'bool' if have == 1 else f'u{have}'} where"
                f" {'bool' if w == 1 else f'u{w}'} is needed",
            )
            return K(None, w)
        if have > w:
            self.error(
                line, f"width error: u{have} value where u{w} is needed (use lo())"
            )
            return K(None, w)
        return zext(expr, w)

    def unify(self, a, b, line: int):
        """Two operands at one width (an untyped constant takes the other's)."""
        wa, wb = width(a), width(b)
        if wa == wb:
            return a, b, wa
        if 1 in (wa, wb):
            self.error(line, "width error: truth value mixed with an integer")
            return a, b, max(wa, wb)
        w = max(wa, wb)
        return self.coerce(a, w, line), self.coerce(b, w, line), w

    def truth(self, expr, scope: Scope):
        """A condition: ``expr`` must be a truth value."""
        value = self.expr(expr, scope)
        if width(value) != 1:
            self.error(expr[1], "width error: condition is not a truth value")
            return K(None, 1)
        return value

    def expr(self, node, scope: Scope):
        """Lower an expression node."""
        # pylint: disable=too-many-return-statements,too-many-branches
        tag, line = node[0], node[1]
        if tag == "_v":
            return node[2]
        if tag == "num":
            return K(node[2], 0)
        if tag == "bool":
            return K(int(node[2]), 1)
        if tag == "name":
            return self.name_value(node[2], scope, line)
        if tag == "attr":
            return self.attr_value(node[2], node[3], scope, line)
        if tag == "index":
            return self.index_value(node[2], node[3], scope, line)
        if tag == "inst_attr":
            inst = self.machine_instance(node[2], self.expr(node[3], scope), line)
            return K(None, 8) if inst is None else self.member(inst, node[4], line)
        if tag in ("mem", "mem16"):
            addr = self.coerce(self.expr(node[2], scope), 16, line)
            if tag == "mem":
                return self.rd(addr)
            return word(self.rd(addr), self.rd(mk("add", 16, addr, K(1, 16))))
        if tag == "func":
            return self.func(node[2], node[3], scope, line)
        if tag in ("&&", "||"):
            a, b = self.truth(node[2], scope), self.truth(node[3], scope)
            return mk("land" if tag == "&&" else "lor", 1, a, b)
        if tag == "!":
            return mk("lnot", 1, self.truth(node[2], scope))
        if tag in ("~", "uminus"):
            value = self.expr(node[2], scope)
            w = width(value)
            if w == 1:
                self.error(line, f"width error: {tag} of a truth value")
            if isinstance(value, K) and w == 0:
                if tag == "~":
                    self.error(line, "width error: ~ of an untyped constant")
                return K(None if value.v is None else -value.v, 0)
            return mk("not" if tag == "~" else "neg", w, value)
        a, b = self.expr(node[2], scope), self.expr(node[3], scope)
        if tag in SHIFT:
            untyped = width(a) == 0 and not (isinstance(b, K) and b.w == 0)
            if 1 in (width(a), width(b)) or untyped:
                self.error(line, "width error: shift needs a typed integer operand")
                return K(None, max(width(a), 8))
            w = width(a)
            if w == 0:
                if a.v is None or b.v is None:
                    return K(None, 0)
                return K(a.v << b.v if tag == "<<" else a.v >> b.v, 0)
            return mk(
                SHIFT[tag], w, a, self.coerce(b, w, line) if width(b) else K(b.v, w)
            )
        if tag in ("/", "%"):
            if not (isinstance(a, K) and isinstance(b, K) and a.w == b.w == 0):
                self.error(line, f"{tag} needs static integer operands")
                return K(None, 0)
            if a.v is None or b.v is None or b.v == 0:
                return K(None, 0)
            return K(a.v // b.v if tag == "/" else a.v % b.v, 0)
        a, b, w = self.unify(a, b, line)
        if tag in ARITH:
            if w == 1:
                self.error(line, f"width error: {tag} of truth values")
            if w == 0:
                if a.v is None or b.v is None:
                    return K(None, 0)
                return K(eval_static(tag, a.v, b.v), 0)
            return mk(ARITH[tag], w, a, b)
        if tag in COMPARE:
            op = COMPARE[tag]
            if w == 1 and op not in ("eq", "ne"):
                self.error(line, f"width error: {tag} of truth values")
            if w == 0:
                if a.v is None or b.v is None:
                    return K(None, 1)
                return K(int(eval_compare(tag, a.v, b.v)), 1)
            if op in ("ugt", "uge"):
                return mk("ult" if op == "ugt" else "ule", 1, b, a)
            return mk(op, 1, a, b)
        self.error(line, f"unknown operator {tag}")
        return K(None, 8)

    def name_value(self, name: str, scope: Scope, line: int):
        """The value of a bare name."""
        found = self.lookup(name, scope)
        if found is None:
            self.error(line, f"undefined name {name!r}")
            return K(None, 8)
        kind = found[0]
        if kind == "var":
            return V(found[1], found[2])
        if kind == "static":
            return found[1]
        if kind == "signal":
            return self.signal_read(found[1], found[2], line)
        if kind == "cell":
            return self.read_cell(found[1], None, line)
        self.error(
            line, f"{name} is a stream; read it with next, {name}[k] or {name}.pos"
        )
        return K(None, 8)

    def attr_value(self, left: str, attr: str, scope: Scope, line: int):
        """``left.attr``: a layout field, stream cursor or instance member."""
        if left == "cpu":
            w = CPU_INPUTS.get(attr)
            if w is None:
                self.error(
                    line, f"cpu.{attr}: entry registers are a x y and flags c z n v d i"
                )
                return K(None, 8)
            self.inputs[f"cpu_{attr}"] = w
            return V(f"cpu_{attr}", w)
        if left == "L":
            value = self.layout(attr, line)
            if not isinstance(value, K):
                self.error(line, f"L.{attr} is a list; use len(), member() or for")
                return K(None, 0)
            return value
        found = scope.names.get(left)
        if found is not None and found[0] == "stream":
            if attr not in ("pos", "base"):
                self.error(line, f"stream {left} has no {attr!r} (pos, base)")
                return K(None, 8)
            return V(
                found[1] if attr == "base" else found[2], 16 if attr == "base" else 8
            )
        owner = self.owner_value(left, attr, scope, line)
        if owner is not None:
            return owner
        inst = self.machine_instance(left, None, line)
        return K(None, 8) if inst is None else self.member(inst, attr, line)

    def index_value(self, name: str, index_node, scope: Scope, line: int):
        """``name[index]``: a cell element or a stream byte relative to its cursor."""
        index = self.expr(index_node, scope)
        found = self.lookup(name, scope)
        if found is not None and found[0] == "stream":
            offset = self.coerce(index, 8, line)
            pos = mk("add", 8, V(found[2], 8), offset)
            return self.rd(add16(V(found[1], 16), pos))
        if found is not None and found[0] == "cell":
            return self.read_cell(found[1], index, line)
        if name in self.spec.machines:
            self.error(line, f"{name}[...] needs a member (.signal or .cell)")
            return K(None, 8)
        self.error(line, f"undefined cell {name!r}")
        return K(None, 8)

    def list_value(self, node, scope: Scope):
        """A static list: ``L.<list>`` or a range."""
        if node[0] == "irange":
            lo = self.static(node[2], scope, node[1])
            hi = self.static(node[3], scope, node[1])
            return UNKNOWN_LIST if None in (lo, hi) else tuple(range(lo, hi))
        expr = node[2] if node[0] == "ilist" else node
        if expr[0] == "attr" and expr[2] == "L":
            value = self.layout(expr[3], expr[1])
            if not isinstance(value, K):
                return value
        self.error(node[1], "expected a static list (L.<list> or a..b)")
        return UNKNOWN_LIST

    def func(self, name: str, args, scope: Scope, line: int):
        """A builtin function."""
        # pylint: disable=too-many-return-statements,too-many-branches
        if name in self.inst.procs or name in self.spec.procs:
            self.error(
                line, f"proc {name} returns only as a whole let/assignment value"
            )
            return K(None, 8)
        if name == "len" or name == "member":
            values = self.list_value(args[-1], scope)
            if values is UNKNOWN_LIST:
                return K(None, 0 if name == "len" else 1)
            if name == "len":
                return K(len(values), 0)
            key = self.expr(args[0], scope)
            if not isinstance(key, K):
                self.error(line, "member() needs a static value")
                return K(None, 1)
            return K(None if key.v is None else int(key.v in values), 1)
        if name == "old":
            if not self.allow_old:
                self.error(line, "old() is only allowed in an invariant")
            saved, self.old = self.old, True
            value = self.expr(args[0], scope)
            self.old = saved
            return value
        vals = [self.expr(a, scope) for a in args]
        arity = {
            "lo": 1,
            "hi": 1,
            "u16": 1,
            "u8": 1,
            "neg": 1,
            "sx": 1,
            "zp": 1,
            "word": 2,
            "slt": 2,
            "sle": 2,
            "sgt": 2,
            "sge": 2,
            "bit": 2,
            "ind": 2,
            "rol": 2,
            "ror": 2,
            "ite": 3,
            "owns": 2,
            "carry": (2, 3),
            "borrow": (2, 3),
        }
        want = arity.get(name)
        if want is None:
            self.error(line, f"undefined function {name!r}")
            return K(None, 8)
        if len(vals) not in (want if isinstance(want, tuple) else (want,)):
            self.error(line, f"{name}() takes {want} arguments")
            return K(None, 8)
        return getattr(self, f"fn_{name}")(vals, line)

    def fn_lo(self, vals, line):
        """Low byte of a word."""
        return mk("lo", 8, self.coerce(vals[0], 16, line))

    def fn_hi(self, vals, line):
        """High byte of a word."""
        return mk("hi", 8, self.coerce(vals[0], 16, line))

    def fn_word(self, vals, line):
        """Little-endian word of two bytes."""
        return word(self.coerce(vals[0], 8, line), self.coerce(vals[1], 8, line))

    def fn_u16(self, vals, line):
        """Zero-extension to 16 bits."""
        value = vals[0]
        return (
            mk("zext", 16, value) if width(value) == 1 else self.coerce(value, 16, line)
        )

    def fn_u8(self, vals, line):
        """A truth value (0/1) or constant as a byte."""
        value = vals[0]
        return (
            mk("zext", 8, value) if width(value) == 1 else self.coerce(value, 8, line)
        )

    def fn_neg(self, vals, line):
        """The sign (top) bit."""
        value = vals[0]
        w = width(value)
        if w in (0, 1):
            self.error(line, "width error: neg() needs a typed integer")
            return K(None, 1)
        return mk("ne", 1, mk("and", w, value, K(1 << (w - 1), w)), K(0, w))

    def fn_sx(self, vals, line):
        """Sign-extension of a byte to 16 bits."""
        return mk("sext", 16, self.coerce(vals[0], 8, line))

    def fn_zp(self, vals, line):
        """A zero-page address (8-bit wrap)."""
        value = vals[0]
        if isinstance(value, K) and value.w == 0:
            return K(None if value.v is None else value.v & 0xFF, 0)
        return mk(
            "and", width(value), value, self.coerce(K(0xFF, 0), width(value), line)
        )

    def _signed(self, vals, line, op, swap):
        a, b, w = self.unify(vals[0], vals[1], line)
        if w in (0, 1):
            a, b = self.coerce(a, 8, line), self.coerce(b, 8, line)
        return mk(op, 1, *((b, a) if swap else (a, b)))

    def fn_slt(self, vals, line):
        """Signed a < b."""
        return self._signed(vals, line, "slt", False)

    def fn_sle(self, vals, line):
        """Signed a <= b."""
        return self._signed(vals, line, "sle", False)

    def fn_sgt(self, vals, line):
        """Signed a > b."""
        return self._signed(vals, line, "slt", True)

    def fn_sge(self, vals, line):
        """Signed a >= b."""
        return self._signed(vals, line, "sle", True)

    def fn_bit(self, vals, line):
        """Bit n (static) of a value."""
        n = vals[1]
        if not isinstance(n, K) or n.w != 0:
            self.error(line, "bit() needs a static bit number")
            return K(None, 1)
        w = max(width(vals[0]), 8)
        value = self.coerce(vals[0], w, line)
        return mk(
            "ne",
            1,
            mk("and", w, value, K(None if n.v is None else 1 << n.v, w)),
            K(0, w),
        )

    def fn_ind(self, vals, line):
        """``(zp),y``: the byte at the zero-page pointer plus ``y``."""
        zp = self.coerce(vals[0], 8, line)
        y = self.coerce(vals[1], 8, line)
        lo = self.rd(zext(zp, 16))
        hi = self.rd(zext(mk("add", 8, zp, K(1, 8)), 16))
        return self.rd(add16(word(lo, hi), y))

    def _rotate(self, vals, line, left):
        value, carry = vals
        w = width(value)
        if w in (0, 1) or width(carry) != 1:
            self.error(line, "width error: rol/ror need (integer, truth value)")
            return K(None, max(w, 8))
        bit = mk("zext", w, carry)
        if left:
            return mk("or", w, mk("shl", w, value, K(1, w)), bit)
        return mk("or", w, mk("shr", w, value, K(1, w)), mk("shl", w, bit, K(w - 1, w)))

    def fn_rol(self, vals, line):
        """Rotate left through a carry-in."""
        return self._rotate(vals, line, True)

    def fn_ror(self, vals, line):
        """Rotate right through a carry-in."""
        return self._rotate(vals, line, False)

    def fn_owns(self, vals, line):
        """Whether the owner cell value ``a`` is the id ``b``."""
        a, b, w = self.unify(vals[0], vals[1], line)
        if w in (0, 1):
            a, b = self.coerce(a, 8, line), self.coerce(b, 8, line)
        return mk("own", 1, a, b)

    def fn_ite(self, vals, line):
        """``c ? a : b``."""
        if width(vals[0]) != 1:
            self.error(line, "width error: ite() condition is not a truth value")
        a, b, w = self.unify(vals[1], vals[2], line)
        if w == 0:
            a, b, w = self.coerce(a, 8, line), self.coerce(b, 8, line), 8
        return mk("ite", w, vals[0], a, b)

    def _carry_args(self, vals, line):
        a, b, w = self.unify(vals[0], vals[1], line)
        if w in (0, 1):
            a, b, w = self.coerce(a, 8, line), self.coerce(b, 8, line), 8
        cin = vals[2] if len(vals) == 3 else K(0, 1)
        if width(cin) != 1:
            self.error(line, "width error: carry-in is not a truth value")
            cin = K(0, 1)
        return a, b, w, cin

    def fn_carry(self, vals, line):
        """Carry out of ``a + b (+ c)``."""
        a, b, w, cin = self._carry_args(vals, line)
        total = mk("add", w, a, b)
        return mk(
            "lor",
            1,
            mk("ult", 1, total, a),
            mk("land", 1, cin, mk("eq", 1, total, K((1 << w) - 1, w))),
        )

    def fn_borrow(self, vals, line):
        """Borrow out of ``a - b (- c)``."""
        a, b, _w, cin = self._carry_args(vals, line)
        return mk("lor", 1, mk("ult", 1, a, b), mk("land", 1, cin, mk("eq", 1, a, b)))

    # -- statements ----------------------------------------------------------
    def block(self, stmts, scope: Scope) -> list:
        """Lower a block; statements after a possible return run under ``!returned``."""
        out: list = []
        stmts = list(flat(stmts))
        for i, stmt in enumerate(stmts):
            if stmt[0] == "return":
                out += self.stmt(stmt, scope)
                if i + 1 < len(stmts):
                    self.error(stmts[i + 1][1], "unreachable statement after return")
                return out
            out += self.stmt(stmt, scope)
            if self.ret is not None and returns(stmt) and i + 1 < len(stmts):
                rest = self.block(stmts[i + 1 :], scope)
                out.append(("if", mk("lnot", 1, V(self.ret[0], 1)), rest, []))
                return out
        return out

    def stmt(self, node, scope: Scope) -> list:
        """Lower one statement."""
        handler = getattr(self, f"st_{node[0]}")
        return handler(node, scope)

    def st_assign(self, node, scope):
        """``lval op= expr``."""
        _tag, line, lval, op, rhs = node
        if op == ":=" and rhs[0] == "func" and self.find_proc(rhs[2]):
            tmp, pre = self.call_value(rhs, scope)
            return pre + self.assign(lval, tmp, scope, line)
        value = self.expr(rhs, scope)
        if op != ":=":
            current = self.expr(lval, scope)
            value = self.expr(
                (COMPOUND[op], line, ("_v", line, current), ("_v", line, value)), scope
            )
        return self.assign(lval, value, scope, line)

    def assign(self, lval, value, scope: Scope, line: int) -> list:
        """Store ``value`` to an lvalue node."""
        tag = lval[0]
        if tag == "mem":
            addr = self.coerce(self.expr(lval[2], scope), 16, line)
            return [("wr", addr, self.coerce(value, 8, line))]
        if tag == "index":
            found = self.lookup(lval[2], scope)
            if found is None or found[0] != "cell":
                self.error(line, f"cannot assign to {lval[2]}[...]")
                return []
            return self.write_cell(found[1], self.expr(lval[3], scope), value, line)
        if tag == "attr":
            found = scope.names.get(lval[2])
            if found is not None and found[0] == "stream" and lval[3] == "pos":
                return [("set", found[2], self.coerce(value, 8, line))]
            if lval[2] in self.spec.machines:
                self.error(
                    line,
                    f"{self.inst.ident or 'schedule'} cannot write {lval[2]}.{lval[3]}"
                    " (signals and cells are written by their own instance)",
                )
                return []
            self.error(line, f"cannot assign to {lval[2]}.{lval[3]}")
            return []
        found = self.lookup(lval[2], scope)
        if found is None:
            self.error(line, f"undefined name {lval[2]!r}")
            return []
        if found[0] == "var":
            return [("set", found[1], self.coerce(value, found[2], line))]
        if found[0] == "signal":
            ir, w = found[1].signals[found[2]]
            return [("set", ir, self.coerce(value, w, line))]
        if found[0] == "cell":
            return self.write_cell(found[1], None, value, line)
        self.error(line, f"cannot assign to {lval[2]}")
        return []

    def st_let(self, node, scope):
        """``let name[: type] = expr``."""
        _tag, line, name, typ, rhs = node
        if rhs[0] == "func" and self.find_proc(rhs[2]):
            value, pre = self.call_value(rhs, scope)
        else:
            value, pre = self.expr(rhs, scope), []
        w = WIDTHS.get(typ) if typ else width(value)
        if typ == "ptr":
            self.error(line, "ptr is not a local type (use u16)")
        if w == 0:
            w = 8 if value.v is None or 0 <= value.v < 256 else 16
        ir = self.fresh(f"l_{name}")
        value = self.coerce(value, w, line)
        scope.names[name] = ("var", ir, w)
        return pre + [("set", ir, value)]

    def stream_of(self, name: str, scope: Scope, line: int):
        """The ``(base, pos)`` IR names of stream ``name``."""
        found = scope.names.get(name)
        if found is None or found[0] != "stream":
            self.error(line, f"{name} is not a stream")
            return None
        return found[1], found[2]

    def next_byte(self, name: str, scope: Scope, line: int):
        """``(value, statements)`` of ``next name``: read at the cursor, then advance."""
        stream = self.stream_of(name, scope, line)
        if stream is None:
            return K(None, 8), []
        base, pos = stream
        tmp = self.fresh("nx")
        return V(tmp, 8), [
            ("set", tmp, self.rd(add16(V(base, 16), V(pos, 8)))),
            ("set", pos, mk("add", 8, V(pos, 8), K(1, 8))),
        ]

    def st_next_let(self, node, scope):
        """``let name = next stream``."""
        value, pre = self.next_byte(node[3], scope, node[1])
        ir = self.fresh(f"l_{node[2]}")
        scope.names[node[2]] = ("var", ir, 8)
        return pre + [("set", ir, value)]

    def st_next_set(self, node, scope):
        """``lval := next stream``."""
        _tag, line, lval, op, name = node
        if op != ":=":
            self.error(line, f"next only assigns with :=, not {op}")
        value, pre = self.next_byte(name, scope, line)
        return pre + self.assign(lval, value, scope, line)

    def st_if(self, node, scope):
        """``if``/``elif``/``else`` (static conditions fold)."""
        _tag, _line, arms, other = node
        if not arms:
            return self.block(other, scope.child())
        (cond, body), rest = arms[0], arms[1:]
        test = self.truth(cond, scope)
        branch_writers = dict((k, set(v)) for k, v in self.writers.items())
        if isinstance(test, K) and test.v is not None:
            return (
                self.block(body, scope.child())
                if test.v
                else self.st_if(("if", node[1], rest, other), scope)
            )
        then = self.block(body, scope.child())
        after_then, self.writers = self.writers, branch_writers
        orelse = self.st_if(("if", node[1], rest, other), scope)
        for key, names in after_then.items():
            self.writers.setdefault(key, set()).update(names)
        return [("if", test, then, orelse)]

    def st_while(self, node, scope):
        """``while cond bound n``."""
        _tag, line, cond, bound, body = node
        test = self.truth(cond, scope)
        inner = self.block(body, scope.child())
        if self.ret is not None and any(returns(s) for s in body):
            test = mk("land", 1, test, mk("lnot", 1, V(self.ret[0], 1)))
        if bound <= 0:
            self.error(line, "while bound must be positive")
        return [("loop", test, inner, bound, line)]

    def st_for(self, node, scope):
        """``for v in list { }``: unrolled over a static list."""
        _tag, line, name, iterable, body = node
        values = self.list_value(iterable, scope)
        out = []
        if values is UNKNOWN_LIST:
            values = (None,)
        for value in values:
            inner = scope.child()
            inner.names[name] = ("static", K(value, 0))
            out += self.block(body, inner)
        del line
        return out

    def st_return(self, node, scope):
        """``return [expr]``."""
        _tag, line, value = node
        if self.ret is None:
            self.error(line, "return outside a proc")
            return []
        flag, slot, w = self.ret
        out = [("set", flag, K(1, 1))]
        if (value is None) != (slot is None):
            self.error(line, "return value does not match the proc's declaration")
        elif value is not None:
            out.insert(0, ("set", slot, self.coerce(self.expr(value, scope), w, line)))
        return out

    def st_assert(self, node, scope):
        """``assert cond``."""
        return [("assert", self.truth(node[2], scope), f"assert at line {node[1]}")]

    def st_stream(self, node, scope):
        """``stream s = base @ cursor [grammar "..."]``."""
        _tag, line, name, base, pos, grammar = node
        base_v = self.coerce(self.expr(base, scope), 16, line)
        pos_v = self.coerce(self.expr(pos, scope), 8, line)
        b, p = self.fresh(f"{name}_base"), self.fresh(f"{name}_pos")
        scope.names[name] = ("stream", b, p)
        out = [("set", b, base_v), ("set", p, pos_v)]
        chosen = self.select_grammar(grammar, scope, line)
        if chosen is not None:
            self.grammars.append((chosen, line))
            out.append(("hook", len(self.grammars) - 1, V(b, 16)))
        return out

    def st_native(self, node, scope):
        """``native addr(a = e, x = e, y = e) resumes pc``: run the tune's own 6502
        code from ``addr``; the spec goes on if it reaches ``pc``, else the CPU
        finishes the call."""
        _tag, line, addr, args, resume, span = node
        regs = []
        for reg, value in args:
            if reg not in ("a", "x", "y") or any(reg == r for r, _ in regs):
                self.error(line, "native arguments are a, x and y, each at most once")
                continue
            regs.append((reg, self.coerce(self.expr(value, scope), 8, line)))
        addr_v = self.coerce(self.expr(addr, scope), 16, line)
        resume_v = self.coerce(self.expr(resume, scope), 16, line)
        self.natives.append((span, addr_v, resume_v))
        return [("native", addr_v, tuple(regs), resume_v)]

    def st_finish(self, node, scope):
        """``finish``: the play call ends here (the rest of the schedule is skipped)."""
        del node, scope
        return [("finish",)]

    def select_grammar(self, alts, scope, line: int) -> Optional[str]:
        """The rule of the first ``"ref" if cond`` arm that holds (an arm with no
        condition always holds); ``cond`` is static, so the layout picks the rule."""
        chosen = None
        for ref, cond in alts:
            if ref.count(":") != 2:
                self.error(line, 'grammar is "module:attribute:rule"')
            if chosen is None:
                if cond is None:
                    chosen = ref
                else:
                    value = self.static(cond, scope, line)
                    if value is None and self.binding is None:
                        chosen = ref  # unbound: every arm is a candidate
                    elif value:
                        chosen = ref
        return chosen

    def control(self, name: str, line: int):
        """The control declaration ``name`` of the current instance."""
        found = self.inst.controls.get(name)
        if found is None:
            self.error(line, f"undefined control {name!r}")
        return found

    def control_value(self, ctl):
        """The masked current state value of a control."""
        kind, ref, mask, _values = ctl
        if kind == "hidden":
            return V(ref, 8)
        cell = self.inst.cells.get(ref) or self.shared[ref]
        return mk("and", 8, self.read_cell(cell, None, 0), K(mask, 8))

    def st_in(self, node, scope):
        """``in control { state => {...} }``: dispatch on the control state."""
        _tag, line, name, arms = node
        ctl = self.control(name, line)
        if ctl is None:
            return []
        current = self.control_value(ctl)
        chain = []
        for state, body in arms:
            if state not in ctl[3]:
                self.error(line, f"control {name} has no state {state!r}")
                continue
            chain.append(((mk("eq", 1, current, K(ctl[3][state], 8))), body))
        out: list = []
        for test, body in reversed(chain):
            then = self.block(body, scope.child())
            out = [("if", test, then, out)]
        return out

    def st_goto(self, node, scope):
        """``goto state``: write the state into its control (cell bits or hidden)."""
        del scope
        _tag, line, state = node
        owners = [c for c in self.inst.controls.values() if state in c[3]]
        if len(owners) != 1:
            self.error(
                line,
                f"goto {state}: {'no' if not owners else 'ambiguous'} control state",
            )
            return []
        kind, ref, mask, values = owners[0]
        if kind == "hidden":
            return [("set", ref, K(values[state], 8))]
        cell = self.inst.cells.get(ref) or self.shared[ref]
        value = mk(
            "or",
            8,
            mk("and", 8, self.read_cell(cell, None, line), K(mask ^ 0xFF, 8)),
            K(values[state] & mask, 8),
        )
        return self.write_cell(cell, None, value, line)

    def find_proc(self, name: str):
        """The proc ``name`` visible from the current instance, else None."""
        return self.inst.procs.get(name) or self.spec.procs.get(name)

    def st_call(self, node, scope):
        """A proc call statement."""
        _tag, line, name, args = node
        proc = self.find_proc(name)
        if proc is None:
            self.error(line, f"undefined proc {name!r}")
            return []
        return self.inline(proc, args, scope, line, None)

    def call_value(self, node, scope):
        """``(value, statements)`` of a value-returning proc call."""
        proc = self.find_proc(node[2])
        if proc[4] is None:
            self.error(node[1], f"proc {node[2]} returns no value")
            return K(None, 8), self.inline(proc, node[3], scope, node[1], None)
        w = WIDTHS[proc[4]]
        slot = self.fresh(f"r_{node[2]}")
        return V(slot, w), [("set", slot, K(0, w))] + self.inline(
            proc, node[3], scope, node[1], slot
        )

    def inline(self, proc, args, scope: Scope, line: int, slot: Optional[str]) -> list:
        """A call of ``proc`` with ``args`` (streams pass by reference).

        The body is lowered once per instance, stream arguments and set of
        instances already run, into :attr:`procs`; each call site only sets the
        parameters, calls it and copies the result to ``slot``.
        """
        _tag, _pline, name, params, _ret, _body = proc
        key = f"{self.inst.ident}.{name}"
        if key in self.stack:
            self.error(line, f"recursive call of {name}")
            return []
        if len(args) != len(params):
            self.error(
                line, f"{name}() takes {len(params)} arguments, {len(args)} given"
            )
            return []
        is_global = name in self.spec.procs and name not in self.inst.procs
        inner = Scope()
        values, streams = [], []
        for (pname, ptype), arg in zip(params, args):
            if ptype == "stream":
                if arg[0] != "name" or self.stream_of(arg[2], scope, line) is None:
                    self.error(line, f"{name}(): {pname} needs a stream")
                    continue
                inner.names[pname] = scope.names[arg[2]]
                streams.append((pname, repr(inner.names[pname])))
                continue
            values.append(
                (
                    pname,
                    WIDTHS[ptype],
                    self.coerce(self.expr(arg, scope), WIDTHS[ptype], line),
                )
            )
        owner = self.global_inst if is_global else self.inst
        cache = (owner.ident, name, slot is None, tuple(streams), frozenset(self.ran))
        if cache not in self.lowered:
            self.lowered[cache] = self.lower_proc(
                proc, key, owner, inner, values, slot is not None
            )
        pid, names, result, writes, reads = self.lowered[cache]
        for cell, rline in reads:
            self.check_latched(cell, rline)
        for cell_key, idents in writes.items():
            self.writers.setdefault(cell_key, set()).update(idents)
        if self.latched is not None:
            self.latched.extend(reads)
        out = [("set", names[pname], value) for pname, _w, value in values]
        out.append(("call", pid))
        if slot is not None:
            out.append(("set", slot, V(result[0], result[1])))
        return out

    def lower_proc(self, proc, key: str, owner, inner: Scope, values, valued: bool):
        """``(id, parameter names, result, writes, latched reads)`` of ``proc`` lowered
        once into :attr:`procs`."""
        _tag, _pline, name, _params, ret, body = proc
        names = {}
        for pname, w, _value in values:
            names[pname] = self.fresh(f"a_{pname}")
            inner.names[pname] = ("var", names[pname], w)
        flag = self.fresh(f"ret_{name}")
        w = WIDTHS.get(ret, 8) if ret else 0
        result = (self.fresh(f"r_{name}"), w) if valued else (None, w)
        saved = self.ret, self.inst, self.latched
        proc_saved, self.proc = self.proc, name
        before = {k: set(v) for k, v in self.writers.items()}
        self.ret = (flag, result[0], w)
        self.inst = owner
        self.latched = []
        self.stack.append(key)
        lowered = self.block(body, inner)
        self.stack.pop()
        reads = self.latched
        self.ret, self.inst, self.latched = saved
        self.proc = proc_saved
        writes = {
            k: new - before.get(k, set())
            for k, new in self.writers.items()
            if new - before.get(k, set())
        }
        prologue = [("set", flag, K(0, 1))] if any(returns(s) for s in body) else []
        if valued:
            prologue.append(("set", result[0], K(0, w)))
        pid = f"{key}#{len(self.procs)}"
        self.procs[pid] = prologue + lowered
        return pid, names, result, writes, reads

    # -- schedule ------------------------------------------------------------
    def run_instance(self, inst: Instance, line: int) -> list:
        """The inlined ``step`` of ``inst`` (one invocation in the frame order)."""
        step = inst.procs.get("step")
        if step is None:
            self.error(line, f"machine {inst.machine[2]} has no step")
            return []
        saved, self.inst = self.inst, inst
        body = self.inline(step, [], Scope(), line, None)
        self.inst = saved
        if "?" not in inst.ident:
            self.ran.add(inst.ident)
        self.order.append(inst.ident)
        return body

    def sitems(self, items, scope: Scope) -> list:
        """Lower schedule items in order, tracking the instances that have run."""
        out = []
        for item in items:
            tag, line = item[0], item[1]
            if tag == "inst":
                index = None if item[3] is None else self.expr(item[3], scope)
                inst = self.machine_instance(item[2], index, line)
                if inst is not None:
                    out += self.run_instance(inst, line)
            elif tag == "sif":
                test = self.truth(item[2], scope)
                if isinstance(test, K) and test.v is not None:
                    out += self.sitems(item[3] if test.v else item[4], scope)
                    continue
                before = set(self.ran)
                then = self.sitems(item[3], scope.child())
                ran_then, self.ran = self.ran, set(before)
                orelse = self.sitems(item[4], scope.child())
                self.ran &= ran_then
                out.append(("if", test, then, orelse))
            else:
                values = self.list_value(item[3], scope)
                before = set(self.ran)
                for value in (None,) if values is UNKNOWN_LIST else values:
                    inner = scope.child()
                    inner.names[item[2]] = ("static", K(value, 0))
                    out += self.sitems(item[4], inner)
                if values is UNKNOWN_LIST:
                    self.ran = before
        return out

    def frame(self) -> list:
        """The whole play call: signal defaults, then the schedule."""
        if not self.spec.schedule:
            self.error(0, "spec has no schedule")
            return []
        body = self.sitems(self.spec.schedule[2], Scope())
        return [("set", ir, K(default, w)) for ir, w, default in self.signals] + body

    def cells(self) -> List[Cell]:
        """Every bound cell (shared, then each instance's)."""
        out = list(self.shared.values())
        for inst in self.instances.values():
            out += inst.cells.values()
        return out + list(self.variants.values())

    def bind_checks(self) -> List[str]:
        """Bind-time checks: private cells unique per instance; notes on aliasing."""
        notes = []
        for name, machine in self.spec.machines.items():
            if machine[3] is None:
                continue
            lo = self.static(machine[4], Scope(), machine[1])
            hi = self.static(machine[5], Scope(), machine[1])
            if None in (lo, hi):
                continue
            insts = [self.instance(name, i) for i in range(lo, hi)]
            for cname in insts[0].cells if insts else ():
                seen: Dict[int, int] = {}
                for i, inst in zip(range(lo, hi), insts):
                    for addr in inst.cells[cname].extent():
                        if seen.setdefault(addr, i) != i:
                            self.error(
                                inst.cells[cname].line,
                                f"private cell {cname} of {name}[{seen[addr]}] and"
                                f" {name}[{i}] share ${addr:04x}",
                            )
        cells = self.cells()
        readonly = {a: c for c in cells if "ro" in c.attrs for a in c.extent()}
        for cell in cells:
            if "ro" in cell.attrs:
                continue
            hits = sorted({readonly[a].name for a in cell.extent() if a in readonly})
            if hits:
                notes.append(
                    f"{cell.owner or 'shared'}.{cell.name} aliases read-only {', '.join(hits)}"
                )
        return notes


def eval_static(tag: str, a: int, b: int) -> int:
    """An untyped static ``a tag b``."""
    return {"+": a + b, "-": a - b, "*": a * b, "&": a & b, "|": a | b, "^": a ^ b}[tag]


def eval_compare(tag: str, a: int, b: int) -> bool:
    """An untyped static comparison."""
    return {
        "==": a == b,
        "!=": a != b,
        "<": a < b,
        "<=": a <= b,
        ">": a > b,
        ">=": a >= b,
    }[tag]


def flat(stmts):
    """``stmts`` with labels stripped and groups spliced in (a group opens no scope)."""
    for stmt in stmts:
        if stmt[0] == "label":
            yield from flat([stmt[3]])
        elif stmt[0] == "group":
            yield from flat(stmt[2])
        else:
            yield stmt


def returns(node) -> bool:
    """Whether a statement node may execute a ``return`` (not counting inlined calls)."""
    tag = node[0]
    if tag in ("label", "group"):
        return any(returns(s) for s in flat([node]))
    if tag == "return":
        return True
    if tag == "if":
        return any(any(returns(s) for s in body) for _c, body in node[2]) or any(
            returns(s) for s in node[3]
        )
    if tag in ("while", "for") or tag == "fire" and node[3] is not None:
        return any(returns(s) for s in node[-1])
    if tag == "in":
        return any(any(returns(s) for s in body) for _n, body in node[3])
    return False


def elaborate(
    spec: Spec,
    binding: Optional[dict] = None,
    routine: Optional[str] = None,
    standard: Optional[int] = None,
) -> Program:
    """Check ``spec`` and lower it: the frame, or one ``routine`` (``"voice[0].parse"``).

    Raises :class:`SpecError` listing every diagnostic. ``binding`` maps the layout
    block's names to values; None checks the spec without one. With ``standard`` (a
    clock in Hz) every read of a cell declared ``notes(...)`` is the standard word of
    its note at that clock rather than memory.
    """
    low = Lowerer(spec, binding, standard)
    if routine is None:
        body = low.frame()
    else:
        body = _routine(low, routine)
    low.ran = set(low.instances)
    low.allow_old = True
    invariants = [(node[2], low.truth(node[3], Scope())) for node in spec.invariants]
    low.allow_old = False
    assumes = [low.truth(node[2], Scope()) for node in spec.assumes]
    notes = low.bind_checks() if binding is not None else []
    low.raise_errors()
    return Program(
        name=spec.name if routine is None else f"{spec.name}:{routine}",
        body=body,
        signals=low.signals,
        controls=low.controls,
        cells=low.cells(),
        invariants=invariants,
        assumes=assumes,
        grammars=low.grammars,
        notes=notes,
        order=low.order,
        inputs=low.inputs,
        natives=low.natives,
        standard=standard,
        procs=low.procs,
        objects=low.all_objects(),
        events=low.events,
    )


def _routine(low: Lowerer, routine: str) -> list:
    """One instance proc, inlined (its signals start at their defaults)."""
    target, _, proc = routine.rpartition(".")
    name, _, index = target.partition("[")
    inst = low.machine_instance(
        name, K(int(index.rstrip("]")), 0) if index else None, 0
    )
    if inst is None or proc not in inst.procs:
        low.error(0, f"no routine {routine!r}")
        low.raise_errors()
    low.ran = set(low.instances) | {inst.ident}
    low.inst = inst
    body = low.inline(inst.procs[proc], [], Scope(), 0, None)
    low.inst = low.global_inst
    return [("set", ir, K(default, w)) for ir, w, default in low.signals] + body


__all__ = ["Lowerer", "elaborate", "mk"]
