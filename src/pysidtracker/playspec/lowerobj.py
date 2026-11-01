"""Lowering of events, records and event-driven objects (a mixin of the lowerer)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .ir import Cell, K, Obj, V
from .lowerops import mk

EVENTS = {
    "frame": None,
    "tick": "frame",
    "row": "tick",
    "note": "row",
    "legato": "row",
    "instrument": "row",
}
SIZES = {"u8": 1, "u16": 2}
TOP = 0xFFFF
MAX_BYTES = 16
Asts = Dict[str, list]


def mentions(node, names) -> bool:
    """Whether the syntax ``node`` names any of ``names``."""
    if isinstance(node, tuple):
        if len(node) > 2 and node[0] == "name" and node[2] in names:
            return True
        return any(mentions(n, names) for n in node[2:])
    return isinstance(node, list) and any(mentions(n, names) for n in node)


def const(value: int, w: int) -> tuple:
    """A syntax node for the constant ``value`` of width ``w``."""
    return ("_v", 0, K(value, w))


class ObjectLowering:
    """Events, records and objects of a :class:`~.lower.Lowerer`."""

    # pylint: disable=too-many-public-methods

    def bind_events(self) -> None:
        """Declare the built-in events and the spec's."""
        self.events = dict(EVENTS)
        for _tag, line, name, parent in self.spec.events:
            if name in self.events:
                self.error(line, f"event {name} is already declared")
            elif parent not in self.events:
                self.error(line, f"event {name}: no event {parent!r}")
            else:
                self.events[name] = parent

    def known_event(self, name: str, line: int) -> bool:
        """Whether ``name`` is an event (else a diagnostic)."""
        if name not in self.events:
            self.error(line, f"undefined event {name!r}")
        return name in self.events

    def below(self, event: str, ancestor: str) -> bool:
        """Whether ``event`` is a descendant of ``ancestor`` (not ``ancestor`` itself)."""
        while self.events.get(event) is not None:
            event = self.events[event]
            if event == ancestor:
                return True
        return False

    def bind_record(self, item, cells: Dict[str, Cell], scope) -> None:
        """Declare the fields of a ``record`` as cells of ``cells``."""
        _tag, line, rec, base, stride, fields = item
        size = self.static(stride, scope, line)
        index = self.inst.machine[3] if self.inst.machine else None
        at = base
        if index is not None:
            at = ("+", line, base, ("*", line, ("name", line, index), stride))
        taken: Dict[int, str] = {}
        offset = 0
        for _t, fline, name, typ, where in fields:
            if where is not None:
                offset = self.static(where, scope, fline) or 0
            if typ not in SIZES:
                self.error(fline, f"record field {name}: u8 or u16 only")
                continue
            end = offset + SIZES[typ]
            if size is not None and end > size:
                self.error(fline, f"record field {name} ends past the stride {size}")
            for byte in range(offset, end):
                if taken.setdefault(byte, name) != name:
                    self.error(fline, f"record field {name} overlaps {taken[byte]}")
            if name in cells:
                self.error(fline, f"record field {name} is already a cell")
            place = ("at", fline, ("+", fline, at, ("_v", fline, K(offset, 0))))
            decl = ("cell", fline, name, typ, None, place, frozenset(), None)
            cells[name] = self.bind_cell(decl, self.inst.ident, scope)
            cells[name].record = rec
            offset = end

    def object_cell(self, name: str, line: int, scope, wide: int = 2) -> tuple:
        """The byte addresses of the scalar cell ``name`` (at most ``wide`` bytes)."""
        found = self.lookup(name, scope)
        cell = found[1] if found and found[0] == "cell" else None
        if cell is None or cell.length is not None or SIZES.get(cell.typ, 9) > wide:
            self.error(line, f"{name} is not a scalar cell of at most {wide} bytes")
            return (None,)
        if cell.typ == "u8":
            return (cell.place[0],)
        if len(cell.place) == 2:
            return cell.place
        low = cell.place[0]
        return (low, None if low is None else (low + 1) & TOP)

    def declare_reacting(self, item, table: Dict[str, object], scope) -> None:
        """Bind an object or record a handler, in the declaration order a firing follows."""
        if item[0] != "handler":
            table[item[2]] = self.bind_object(item, scope)
            return
        for event in item[3]:
            self.known_event(event, item[1])
        if item[2] in self.inst.objects or item[2] in self.shared_objs:
            self.error(item[1], f"{item[2]} is already declared")
        table[item[2]] = item

    def bind_object(self, item, scope) -> Optional[Obj]:
        """The :class:`~.ir.Obj` an object declaration binds in the current instance."""
        tag, line, name = item[0], item[1], item[2]
        if name in self.inst.objects or name in self.shared_objs:
            self.error(line, f"object {name} is already declared")
        ident = self.inst.ident
        oid = f"{ident}.{name}" if ident else name
        obj, asts = getattr(self, f"bind_{tag}")(item, oid, scope)
        obj.owner, obj.line = ident, line
        self.asts[oid] = asts
        return obj

    def reaction(self, name: str, events, line: int, table: dict, action) -> None:
        """Record ``action`` as the reaction to each of ``events`` (once each)."""
        for event in events:
            if self.known_event(event, line):
                if event in table:
                    self.error(line, f"{name} reacts to {event} twice")
                table[event] = action

    def bind_counter(self, item, oid, scope) -> Tuple[Obj, Asts]:
        """``counter name: cell { on ev += k; wrap [at e] [-> ev]; reload e; }``."""
        _tag, line, name, cell, clauses = item
        loc = self.object_cell(cell, line, scope)
        w = 8 * len(loc)
        reactions: dict = {}
        limit = reload = fires = None
        for clause in clauses:
            if clause[0] == "c_on":
                _t, cline, event, op, amount = clause
                action = ("reset",)
                if op != "reset":
                    step = self.static(amount, scope, cline)
                    if op not in ("+=", "-=") or not step or step < 0:
                        self.error(cline, "a counter steps by a positive constant")
                    action = ("up" if op == "+=" else "down", step or 0)
                self.reaction(name, event, cline, reactions, action)
            elif clause[0] == "c_wrap":
                limit, fires = clause[2], clause[3]
                if fires is not None:
                    self.known_event(fires, clause[1])
            else:
                reload = clause[2]
        for event, action in reactions.items():
            if fires and action[0] != "reset" and not self.below(fires, event):
                self.error(line, f"{name} fires {fires}, which is not below {event}")
                fires = None
        asts = [limit or const((1 << w) - 1, w), reload or const(0, w)]
        obj = Obj(oid, "counter", "", {"cell": loc}, (), reactions, fires)
        return obj, dict.fromkeys(reactions, asts)

    def bind_owner(self, item, oid, scope) -> Tuple[Obj, Asts]:
        """``owner name: cell { id e; none e; on ev claim|release [when e]; }``."""
        _tag, line, name, cell, clauses = item
        loc = self.object_cell(cell, line, scope, 1)
        ident, none = None, const(0xFF, 8)
        reactions: dict = {}
        when: dict = {}
        for clause in clauses:
            if clause[0] == "o_id":
                ident = clause[2]
            elif clause[0] == "o_none":
                none = clause[2]
            else:
                _t, cline, events, action, cond = clause
                self.reaction(name, events, cline, reactions, (action,))
                when.update(dict.fromkeys(events, cond or const(1, 1)))
        self.owner_exprs[oid] = (ident, none)
        asts = {event: [ident, none, cond] for event, cond in when.items()}
        return Obj(oid, "owner", "", {"cell": loc}, (), reactions), asts

    def xform(self, node, cols: Dict[str, int], scope, line: int):
        """``(column, mask, shl, shr)`` of an operand over the byte names ``cols``: a
        column or a constant, ``& mask`` and one shift."""
        tag = node[0]
        if tag == "name" and node[2] in cols:
            return (cols[node[2]], 0xFF, 0, 0)
        if not mentions(node, cols):
            value = self.static(node, scope, line)
            return None if value is None else (-1, value & 0xFF, 0, 0)
        if tag in ("&", "<<", ">>"):
            left, right = node[2], node[3]
            if tag == "&" and mentions(right, cols):
                left, right = right, left
            inner = self.xform(left, cols, scope, line)
            amount = self.static(right, scope, line)
            if inner and amount is not None and not inner[2] and not inner[3]:
                if tag == "&":
                    return (inner[0], inner[1] & amount, 0, 0)
                if inner[0] >= 0 and 0 <= amount < 16:
                    shift = (amount, 0) if tag == "<<" else (0, amount)
                    return (inner[0], inner[1], *shift)
        self.error(line, "an operand is a column or constant, `& mask` and one shift")
        return None

    def assigns(self, sets, names, cols, scope, line: int) -> tuple:
        """``((value index, part, xf), ...)`` of ``value[.lo|.hi] = operand`` items."""
        out = []
        for name, part, expr in sets:
            if part not in (None, "lo", "hi") or name not in names:
                self.error(line, f"{name}.{part}: a value of {names} (.lo or .hi)")
                continue
            spec = self.xform(expr, cols, scope, line) or (-1, 0, 0, 0)
            out.append((names.index(name), (None, "lo", "hi").index(part), spec))
        return tuple(out)

    def bind_walker(self, item, oid, scope) -> Tuple[Obj, Asts]:
        """A walker: table, position and sweep counter cells, values, reactions, rows."""
        _tag, line, name, clauses = item
        by_tag: Dict[str, list] = {}
        for clause in clauses:
            by_tag.setdefault(clause[0], []).append(clause)
        for need in ("w_table", "w_pos", "w_count", "w_values"):
            if len(by_tag.get(need, ())) != 1:
                self.error(line, f"walker {name} needs exactly one {need[2:]} clause")
                return Obj(oid, "walker", "", {}, (1, 1, ()), {}), {}
        _t, tline, bases, width = by_tag["w_table"][0]
        names = by_tag["w_values"][0][2]
        locs = {
            "pos": self.object_cell(by_tag["w_pos"][0][2], line, scope),
            "cnt": self.object_cell(by_tag["w_count"][0][2], line, scope, 1),
        }
        for i, value in enumerate(names):
            locs[f"v{i}"] = self.object_cell(value, line, scope)
        step = None if width is None else self.static(width, scope, tline)
        columns = step if len(bases) == 1 and step else len(bases)
        if len(bases) == 1 and columns > 1:
            bases = [("+", tline, bases[0], const(j, 0)) for j in range(columns)]
        cols = {f"c{j}": j for j in range(columns)}
        rows = tuple(
            self.walker_row(c, names, cols, scope) for c in by_tag.get("w_row", ())
        )
        reactions: dict = {}
        for _t, cline, event, action in by_tag.get("w_on", ()):
            self.reaction(name, event, cline, reactions, (action,))
        start = by_tag["w_start"][0][2] if "w_start" in by_tag else const(0, 8)
        self.check_walker(name, line, locs, columns, rows)
        spec = (step or 1, columns, rows)
        obj = Obj(oid, "walker", "", locs, spec, reactions)
        return obj, dict.fromkeys(reactions, [*bases, start])

    def check_walker(
        self, name: str, line: int, locs: dict, columns: int, rows
    ) -> None:
        """Diagnose rows that name a high byte or fraction their value lacks."""
        if not 0 < columns <= MAX_BYTES:
            self.error(line, f"walker {name} reads 1..{MAX_BYTES} columns")
        for _lo, _hi, action in rows:
            sets = ()
            if action[0] in ("sweep", "delay", "set"):
                sets = (*action[-1], *(action[1] if action[0] == "set" else ()))
            for index, part, _xf in sets:
                if part == 2 and len(locs[f"v{index}"]) < 2:
                    self.error(line, f"walker {name}: value {index} has no high byte")
            if action[0] == "sweep":
                frac, wide = action[4], len(locs[f"v{action[1]}"]) == 2
                if frac and not (wide and 0 < frac <= 24):
                    self.error(line, f"walker {name}: frac needs a word and 1..24")

    def walker_row(self, clause, names, cols, scope) -> tuple:
        """``(lo, hi, action)`` of a ``row`` clause."""
        _t, line, lo, hi, act, then = clause
        low = self.static(lo, scope, line) or 0
        high = low if hi is None else self.static(hi, scope, line) or 0
        after = self.assigns(then, names, cols, scope, line)
        kind = act[0]
        args = {a[0]: a[2] for a in act[2]} if kind in ("sweep", "delay") else {}
        if kind == "sweep":
            into = args.get("into")
            if not {"into", "count", "step"} <= set(args) or into[0] != "name":
                self.error(line, "sweep(into=, count=, step=[, frac=])")
                return (low, high, ("stop",))
            index = names.index(into[2]) if into[2] in names else 0
            frac = self.static(args["frac"], scope, line) if "frac" in args else 0
            count, delta = (
                self.xform(args[k], cols, scope, line) for k in ("count", "step")
            )
            action = ("sweep", index, count, delta, frac or 0, after)
        elif kind == "delay":
            action = ("delay", self.xform(act[2][0][2], cols, scope, line), after)
        elif kind == "set":
            action = ("set", self.assigns(act[2], names, cols, scope, line), after)
        elif kind == "jump":
            action = ("jump", self.xform(act[2], cols, scope, line))
        else:
            action = ("stop",)
        return (low, high, action)

    def condition(self, node, prev: str, scope, line: int):
        """``(mask, value, equal)``: the next field is there when ``(byte & mask) ==
        value`` is ``equal`` of the byte named ``prev``."""
        tag = node[0]
        if tag == "!":
            inner = self.condition(node[2], prev, scope, line)
            return inner and (inner[0], inner[1], not inner[2])
        if tag == "func" and node[2] in ("neg", "bit") and node[3]:
            first = node[3][0]
            if first[0] == "name" and first[2] == prev:
                if node[2] == "neg":
                    return (0x80, 0, False)
                bit = self.static(node[3][1], scope, line)
                return None if bit is None else (1 << bit, 0, False)
        if tag in ("==", "!="):
            left, mask = node[2], 0xFF
            if left[0] == "&" and left[2][0] == "name":
                mask = self.static(left[3], scope, line)
                left = left[2]
            if left[0] == "name" and left[2] == prev and mask is not None:
                value = self.static(node[3], scope, line)
                return None if value is None else (mask, value, tag == "==")
        self.error(line, f"a field's condition tests {prev}: neg, bit, == or !=")
        return None

    def bind_cursor(self, item, oid, scope) -> Tuple[Obj, Asts]:
        """A cursor over the stream ``base @ pos``: reactions, run marker and fields."""
        _tag, line, name, base, pos, grammar, clauses = item
        locs = {"pos": self.object_cell(pos, line, scope, 1)}
        reactions: dict = {}
        run = None
        fields: list = []
        names: List[str] = []
        for clause in clauses:
            tag, cline = clause[0], clause[1]
            if tag == "k_on":
                self.reaction(name, clause[2], cline, reactions, (clause[3],))
            elif tag == "k_run":
                locs["run"] = self.object_cell(clause[5], cline, scope, 1)
                run = tuple(self.static(x, scope, cline) or 0 for x in clause[2:5])
            else:
                _t, _l, field, when, out, expr = clause
                cond = (0, 0, True)
                if when is not None:
                    got = names and self.condition(when, names[-1], scope, cline)
                    if not names:
                        self.error(cline, "the first field is always there")
                    cond = got or cond
                spec = (0xFF, 0, 0)
                if out is not None:
                    locs[f"o{len(fields)}"] = self.object_cell(out, cline, scope)
                    spec = (self.xform(expr, {field: 0}, scope, cline) or (0, *spec))[
                        1:
                    ]
                fields.append((cond, len(fields) if out else None, *spec))
                names.append(field)
        if not 0 < len(fields) <= MAX_BYTES:
            self.error(line, f"cursor {name} has 1..{MAX_BYTES} fields")
        if run and run[0] < run[2]:
            self.error(line, f"cursor {name}: a run's base is above its range")
        chosen = self.select_grammar(grammar, scope, line) if grammar else None
        index = None
        if chosen is not None:
            self.grammars.append((chosen, line))
            index = len(self.grammars) - 1
        obj = Obj(oid, "cursor", "", locs, (run, tuple(fields)), reactions, None, index)
        return obj, dict.fromkeys(reactions, [base, const(0, 8)])

    def visible_objects(self) -> list:
        """What a firing reaches: the shared objects and handlers, then the current
        instance's, each group in declaration order."""
        return [*self.shared_objs.values(), *self.inst.objects.values()]

    def find_object(self, name: str) -> Optional[Obj]:
        """The object ``name`` visible from the current instance."""
        found = self.inst.objects.get(name) or self.shared_objs.get(name)
        return found if isinstance(found, Obj) else None

    def all_objects(self) -> Dict[str, Obj]:
        """Every bound object by id."""
        tables = [self.shared_objs, *(i.objects for i in self.instances.values())]
        return {o.name: o for t in tables for o in t.values() if isinstance(o, Obj)}

    def operands(self, obj: Obj, event: str, scope, line: int) -> list:
        """The operands of ``obj``'s reaction to ``event``, lowered where it happens."""
        fresh = type(scope)()
        return [
            self.coerce(self.expr(self.given(ast), fresh), w, line)
            for ast, w in zip(self.asts[obj.name][event], obj.widths())
        ]

    def given(self, ast):
        """``ast``, or for an owner's default id the index of the instance reacting."""
        index = self.inst.machine[3] if self.inst.machine else None
        if ast is None:
            return ("name", 0, index) if index else const(0, 8)
        return ast

    def step_stmt(self, obj: Obj, event: str, ops: list, flag: Optional[str]) -> list:
        """The statements of one reaction (a cursor's grammar hook first)."""
        locs = [a for loc in obj.locs.values() for a in loc if a is not None]
        reads = [(a, a) for a in locs]
        windows, extra = 0, 0
        if obj.kind == "walker":
            windows, extra = obj.spec[1], 0xFF + obj.spec[0]
        elif obj.kind == "cursor":
            windows, extra = 1, 0xFF + len(obj.spec[1])
        dyn = []
        for index, base in enumerate(ops[:windows]):
            if isinstance(base, K) and base.v is not None and base.v + extra <= TOP:
                reads.append((base.v, base.v + extra))
            else:
                dyn.append((index, extra))
        step = [
            (
                "obj",
                obj.name,
                event,
                flag,
                tuple(ops),
                tuple(reads),
                tuple(locs),
                tuple(dyn),
            )
        ]
        if obj.grammar is not None and obj.reactions[event] == ("next",):
            return [("hook", obj.grammar, ops[0])] + step
        return step

    def reactions_to(self, event: str, scope, line: int) -> list:
        """The statements reacting to ``event``: each visible object's reaction and, for a
        counter that wraps, the firing of its event."""
        out: list = []
        for obj in self.visible_objects():
            if isinstance(obj, tuple):
                if event in obj[3]:
                    out += self.block(obj[4], type(scope)())
                continue
            if event not in obj.reactions:
                continue
            flag = self.fresh("wrap") if obj.fires else None
            ops = self.operands(obj, event, scope, line)
            out += self.step_stmt(obj, event, ops, flag)
            if flag:
                inner = self.reactions_to(obj.fires, scope, line)
                out.append(("if", V(flag, 1), [("fire", obj.fires, inner)], []))
        return out

    def st_fire(self, node, scope):
        """``fire event;`` runs the reactions; ``fire event { ... }`` states them."""
        _tag, line, event, body = node
        if not self.known_event(event, line):
            return []
        if body is not None:
            return [("fire", event, self.block(body, scope.child()))]
        return [("fire", event, self.reactions_to(event, scope, line))]

    def st_react(self, node, scope):
        """``react object on event(operands) [-> flag];``."""
        _tag, line, name, event, args, out = node
        obj = self.find_object(name)
        if obj is None or event not in obj.reactions:
            self.error(line, f"no object {name!r} reacting to {event!r}")
            return []
        if len(args) != obj.arity():
            self.error(line, f"react {name} on {event} takes {obj.arity()} operands")
            return []
        ops = [
            self.coerce(self.expr(a, scope), w, line)
            for a, w in zip(args, obj.widths())
        ]
        flag = None
        if out is not None:
            found = scope.names.get(out)
            if found is None or found[0] != "var" or found[2] != 1:
                self.error(line, f"-> {out} needs a bool variable")
            else:
                flag = found[1]
        return self.step_stmt(obj, event, ops, flag)

    def owner_value(self, name: str, attr: str, scope, line: int):
        """``owner.mine`` (the owner cell holds this instance's id) or ``owner.free``
        (it holds ``none``), if ``name`` is an owner."""
        obj = self.find_object(name)
        if obj is None or obj.kind != "owner":
            return None
        held = self.rd(K(obj.locs["cell"][0], 16))
        ident, none = (
            self.coerce(self.expr(self.given(a), scope), 8, line)
            for a in self.owner_exprs[obj.name]
        )
        if attr in ("mine", "free"):
            return mk(
                "own" if attr == "mine" else "eq",
                1,
                held,
                ident if attr == "mine" else none,
            )
        self.error(line, f"owner {name} has mine and free")
        return K(None, 1)


__all__ = ["ObjectLowering"]
