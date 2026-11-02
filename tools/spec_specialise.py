"""Specialise a playspec's source to one build: every build flag fixed, the branches it cannot
take removed (``if``/``elif``/``else`` blocks, ``ite``, arm and row guards, ``&&``/``||``).

``spec_specialise.py SPEC.play TUNE.sid [--out OUT.play]``: the flags are those the format's
binder gives the tune.
"""

import argparse
import re
import sys
from pathlib import Path

CONST = re.compile(r"!?\b(true|false)\b")
GUARD = r"(?:[!( ]*(?:true|false)[) ]*(?:&&|\|\|)?)+"


def flags_of(tune: Path) -> dict:
    """The tune's layout fields as its format's binder gives them."""
    # pylint: disable=import-outside-toplevel,protected-access
    from pysidtracker import find_format

    raw = tune.read_bytes()
    fmt = find_format(raw)
    player = fmt.spec_player(fmt.parser.parse(raw))
    return player._bind(player.view(), player.entry)


def static(cond: str):
    """True or False when ``cond`` is made of constants alone, else None."""
    text = " ".join(cond.split())
    if not text or set(re.sub(r"\b(true|false)\b", "", text)) - set(" ()!&|"):
        return None
    py = text.replace("&&", " and ").replace("||", " or ").replace("!", " not ")
    return bool(
        eval(py.replace("true", "True").replace("false", "False"))
    )  # pylint: disable=eval-used


def reduce(cond: str):
    """``cond`` with its constant ``&&``/``||`` terms folded: a bool, or the remaining text."""
    cond = " ".join(cond.split())
    whole = static(cond)
    if whole is not None:
        return whole
    while (
        cond.startswith("(")
        and cond.endswith(")")
        and len(split_top(cond[1:-1], "||")) == 1
    ):
        if cond.count("(") != cond[1:-1].count("(") + 1:
            break
        cond = cond[1:-1]
    for op, absorb in (("||", True), ("&&", False)):
        parts = split_top(cond, op)
        if len(parts) > 1:
            kept = []
            for part in parts:
                value = reduce(part)
                if value is absorb:
                    return absorb
                if value is not (not absorb):
                    kept.append(
                        value if op == "&&" or "||" not in value else f"({value})"
                    )
            if not kept:
                return not absorb
            return f" {op} ".join(kept)
    return cond


def split_top(text: str, op: str) -> list:
    """``text`` split at ``op`` outside parentheses."""
    out, depth, start, i = [], 0, 0, 0
    while i < len(text):
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        if depth == 0 and text.startswith(op, i):
            out.append(text[start:i])
            start = i = i + len(op)
            continue
        i += 1
    return out + [text[start:]]


def block_end(text: str, start: int) -> int:
    """The index after the ``}`` closing the ``{`` at ``start``."""
    depth = 0
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return i + 1
    raise ValueError("unbalanced braces")


def dedent(body: str) -> str:
    """A block's body one level out."""
    return "".join(
        line[2:] if line.startswith("  ") else line for line in body.splitlines(True)
    )


IF = re.compile(r"^( *)(?:\} el)?if ([^{]+?) \{\n", re.M)


def ifs(text: str) -> str:
    """``if``/``elif``/``else`` chains with constant conditions reduced to the taken branch."""
    pos = 0
    while (m := IF.search(text, pos)) is not None:
        if text[m.start() : m.start() + len(m[1]) + 1].endswith("}"):
            pos = m.end()
            continue
        indent, arms, at = m[1], [], m.start()
        cond, head = m[2], m.end() - 2
        while True:
            end = block_end(text, head)
            arms.append((cond, text[head + 2 : end - 1].rstrip(" ")))
            nxt = re.match(r" elif ([^{]+?) \{\n| else \{\n", text[end:])
            if nxt is None:
                break
            cond = nxt[1] if nxt[1] is not None else "else"
            head = end + nxt.end() - 2
        kept, taken = [], None
        for cond, body in arms:
            value = cond if cond == "else" else reduce(cond)
            if value is False:
                continue
            if value is True or value == "else":
                taken = body
                break
            kept.append((value, body))
        if not kept:
            new = dedent(taken) if taken else ""
            end += text[end : end + 1] == "\n"
        else:
            new = "".join(
                f"{indent if i == 0 else '} el'}if {c} {{\n{b}{indent}"
                for i, (c, b) in enumerate(kept)
            )
            new += f"}} else {{\n{taken}{indent}}}" if taken is not None else "}"
        text = text[:at] + new + text[end:]
        pos = at if not kept else at + 1
    return text


def guards(text: str) -> str:
    """Arm guards (``pattern if g =>``) and ``then`` item guards (``v = op if g``) with constant
    ``g``: the arm or item dropped when false, the guard dropped when true."""

    def item(m):
        kept = []
        for one in split_top(m[2], ","):
            g = re.search(rf" if ({GUARD})(?= when |$)", one)
            value = None if g is None else reduce(g[1])
            if value is not False:
                kept.append(
                    one
                    if g is None or value is not True
                    else one[: g.start()] + one[g.end() :]
                )
        return (
            f"{m[1]} then {', '.join(k.strip() for k in kept)};" if kept else f"{m[1]};"
        )

    text = re.sub(r"^([^\n]*?) then ([^\n]*);$", item, text, flags=re.M)
    while (m := re.search(rf" if ({GUARD})(?= when | =>)", text)) is not None:
        value = reduce(m[1])
        if value is True:
            text = text[: m.start()] + text[m.end() :]
            continue
        start = text.rfind("\n", 0, m.start()) + 1
        arrow = text.index("=>", m.end())
        body = arrow + 2 + len(text[arrow + 2 :]) - len(text[arrow + 2 :].lstrip(" "))
        end = (
            block_end(text, body) if text[body] == "{" else text.index(";", m.end()) + 1
        )
        text = text[:start] + text[end:].lstrip(" ").removeprefix("\n")
    return text


def ites(text: str) -> str:
    """``ite(c, a, b)`` with ``c`` folded: ``a`` or ``b``, or the reduced condition."""

    def one(m):
        value = reduce(m[1])
        if isinstance(value, bool):
            return m[2] if value else m[3]
        return f"ite({value}, {m[2]}, {m[3]})"

    arg = r"((?:[^(),]|\((?:[^()]|\([^()]*\))*\))+)"
    return re.sub(rf"ite\({arg}, {arg}, {arg}\)", one, text)


def layout(text: str) -> str:
    """The ``layout`` block without fields the rest of the spec no longer names."""
    m = re.search(r"layout \{(.*?)\n\}", text, re.S)
    if m is None:
        return text
    used = set(re.findall(r"L\.(\w+)", text[: m.start()] + text[m.end() :]))
    groups = []
    for names, kind in re.findall(r"([^;:]+):\s*(\w+);", m[1]):
        kept = [n for n in re.findall(r"\w+", names) if n in used]
        if kept:
            groups.append(f"  {', '.join(kept)}: {kind};")
    return (
        text[: m.start()] + "layout {\n" + "\n".join(groups) + "\n}" + text[m.end() :]
    )


def unused(text: str) -> str:
    """``text`` without procs, cells and events nothing else names, until none is left."""
    while True:
        code = re.sub(r"//[^\n]*", "", text)
        drop = None
        for m in re.finditer(r"^( *)(?:proc|event|cell) (\w+)\b", code, re.M):
            if len(re.findall(rf"\b{m[2]}\b", code)) == 1:
                drop = m[2]
                break
        if drop is None:
            return text
        m = re.search(rf"^ *(?:proc|event|cell) {drop}\b", text, re.M)
        start = m.start()
        while text.rfind("\n", 0, start - 1) >= 0 and text[
            text.rfind("\n", 0, start - 1) + 1 : start
        ].lstrip().startswith("//"):
            start = text.rfind("\n", 0, start - 1) + 1
        brace, semi = text.find("{", m.end()), text.find(";", m.end())
        end = block_end(text, brace) if "proc" in m[0] else semi + 1
        text = text[:start] + text[end:].removeprefix("\n")


def specialise(text: str, flags: dict) -> str:
    """``text`` with every build flag in ``flags`` fixed and the untaken branches removed."""
    text = re.sub(
        r"(!?)L\.(\w+)\b",
        lambda m: (
            str(bool(flags[m[2]]) != bool(m[1])).lower() if m[2] in flags else m[0]
        ),
        text,
    )
    text = CONST.sub(
        lambda m: str((m[1] == "true") != m[0].startswith("!")).lower(), text
    )
    return layout(unused(ifs(ites(guards(text)))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("spec", type=Path)
    parser.add_argument("tune", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = args.spec.read_text(encoding="utf-8")
    declared = set(re.findall(r"\w+", "".join(re.findall(r"([^;:]*): flag;", text))))
    flags = {k: v for k, v in flags_of(args.tune).items() if k in declared}
    out = specialise(text, flags)
    if args.out:
        args.out.write_text(out, encoding="utf-8")
    else:
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
