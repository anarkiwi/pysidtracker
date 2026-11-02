"""Lowering of ``state`` arrays: song data and spec-owned rows indexed at run time (a mixin)."""

from __future__ import annotations

from typing import Optional

from .ir import K, V, width
from .lowerops import mk, zext

UNKNOWN_LIST = ("?",)


class ArrayLowering:
    """Array states of a :class:`~.lower.Lowerer`.

    A binding is ``("array", key, w, ro, payload, n)``. A ``ro`` array's payload is its
    values (None while unknown), read through the ``lut`` operator over a constant table. A
    mutable array is ``n`` hidden controls ``key[i]`` (the payload lists their IR names); a
    read of it is the ``aget`` operator over them, a store a binary decision tree of ``set``
    behind an index assertion.
    """

    def bind_array(self, states: dict, item, scope, owner: str = "") -> None:
        """Declare ``state name: T[n] [ro] = list;``."""
        _tag, line, name, typ, initial, size, attr = item
        w = int(typ[1:]) if typ[:1] == "u" else 0
        key = f"{owner}.{name}" if owner else name
        ro = attr == "ro"
        if w == 0 or attr not in (None, "ro"):
            self.error(line, f"state {name}: arrays of u8..u16, optionally `ro`")
            w = w or 8
        want = None if size[2] is None else self.static(size[2], scope, line)
        items, listed = self.array_items(name, initial, ro, scope, line)
        n = want
        values = items
        if items is not UNKNOWN_LIST:
            if listed:
                n = len(items) if want is None else want
                if len(items) != n:
                    self.error(
                        line,
                        f"state {name}: {len(items)} initial items for {n} elements",
                    )
                    values = UNKNOWN_LIST
            elif want is None:
                if size[2] is None:
                    self.error(line, f"state {name}: the length is needed ([n])")
                values = UNKNOWN_LIST
            else:
                values = items * want
        if values is not UNKNOWN_LIST and any(not 0 <= v < 1 << w for v in values):
            self.error(line, f"state {name}: an item does not fit u{w}")
            values = UNKNOWN_LIST
        known = values is not UNKNOWN_LIST
        if ro:
            states[name] = ("array", key, w, True, values if known else None, n)
            return
        elems = tuple(self.fresh(f"state_{name}") for _ in values) if known else ()
        for i, (ir, v) in enumerate(zip(elems, values if known else ())):
            self.state_widths[ir] = w
            self.keys[ir] = f"{key}[{i}]"
            self.controls.append((ir, w, v))
        if elems:
            self.arrays[key] = elems
        states[name] = ("array", key, w, False, elems, n)

    def array_items(self, name: str, initial, ro: bool, scope, line: int):
        """``(items, listed)``: the initial items of an array state, UNKNOWN_LIST while the
        binding is; ``listed`` is False for the one static item every element starts as.
        """
        if initial[0] == "alist":
            items = [self.static(e, scope, line) for e in initial[2]]
            listed = True
        elif (
            initial[0] == "attr"
            and initial[2] == "L"
            and self.spec.layout.get(initial[3]) == "list"
        ):
            items, listed = self.layout(initial[3], line), True
        else:
            if ro:
                self.error(line, f"state {name}: a ro array is initialised from a list")
            items, listed = [self.static(initial, scope, line)], False
        if items is UNKNOWN_LIST or None in items:
            return UNKNOWN_LIST, listed
        return tuple(items), listed

    def array_ref(self, name: str, scope, line: int, ro: Optional[bool] = None):
        """The array binding named ``name`` (None after a diagnostic); ``ro`` True demands
        a read-only byte array, as ``table``, ``view`` and ``cursor`` read."""
        found = self.lookup(name, scope)
        if found is None or found[0] != "array":
            self.error(line, f"{name} is not a state array")
            return None
        if ro and not (found[3] and found[2] == 8):
            self.error(line, f"{name} must be a `u8[] ro` state array")
            return None
        return found

    def array_read(self, arr, index, line: int):
        """``arr[index]``."""
        _kind, key, w, ro, payload, n = arr
        if width(index) == 1:
            self.error(line, "a truth value cannot index")
            return K(None, w)
        if isinstance(index, K) and index.v is not None:
            if n is not None and not 0 <= index.v < n:
                self.error(line, f"{key}[{index.v}] is past its {n} items")
                return K(None, w)
            if payload:
                return K(payload[index.v], w) if ro else V(payload[index.v], w)
        if not payload:
            return K(None, w)
        index = self.array_index(index, n)
        if ro:
            return mk("lut", w, K(payload, w), index)
        return mk("aget", w, index, *(V(e, w) for e in payload))

    @staticmethod
    def array_index(index, n):
        """``index`` as wide as ``n`` items need."""
        index = zext(index, 16) if width(index) == 0 or n > 256 else index
        return index

    def array_write(self, arr, index, value, line: int) -> list:
        """Statements storing ``value`` to ``arr[index]``."""
        _kind, key, w, ro, elems, n = arr
        if ro:
            self.error(line, f"write to read-only array {key}")
            return []
        value = self.coerce(value, w, line)
        if isinstance(index, K) and index.v is not None:
            if n is not None and not 0 <= index.v < n:
                self.error(line, f"{key}[{index.v}] is past its {n} items")
                return []
            return [("set", elems[index.v], value)] if elems else []
        if not elems or width(index) == 1:
            return []
        index = self.array_index(index, n)
        pre = []
        if not isinstance(value, (K, V)):
            tmp = self.fresh("av")
            pre, value = [("set", tmp, value)], V(tmp, w)
        if not isinstance(index, (K, V)):
            tmp = self.fresh("ai")
            pre, index = pre + [("set", tmp, index)], V(tmp, width(index))
        iw = width(index)

        def tree(lo: int, hi: int) -> list:
            if hi - lo == 1:
                return [("set", elems[lo], value)]
            mid = (lo + hi) // 2
            return [
                ("if", mk("ult", 1, index, K(mid, iw)), tree(lo, mid), tree(mid, hi))
            ]

        guard = []
        if n < 1 << iw:
            guard = [
                (
                    "assert",
                    mk("ult", 1, index, K(n, iw)),
                    f"{key}[...] index past its {n} items",
                )
            ]
        return pre + guard + tree(0, n)
