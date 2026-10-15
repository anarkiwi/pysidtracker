"""Run a whole C64 program (editor/packer) headlessly, with KERNAL I/O trapped to the host.

A jennings 6502 over the C64 address space whose file and keyboard routines run in Python, so an
editor's ``SAVE``/``LOAD`` hit ``host_dir`` and ``GETIN`` reads a key queue -- the
offline "run the original editor/packer, no VICE" oracle. :data:`ABI` is the contract.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, Optional, Set, Union

from jennings.devices.mpu6502 import MPU

from . import roms
from .emu import _BankedMemory
from .errors import KernalHostError
from .image import MEM_SIZE, SidImage, parse_prg

ABI = """\
KERNAL call contract: inputs read on the JSR, outputs in the MPU registers; carry
clear on success, set on error. One drive = one host_dir subdirectory; the path is
host_dir/drive/name, name + .seq/.prg/.rel when it lacks a dot; the SETLFS logical
(usually 1/2) keys the open channel, not the path.
  OPEN    --                          A=0 ok/errno; carry (existing file -> input, new -> output)
  CLOSE   A=L                         A=errno; carry           (L=255 closes all)
  CHROUT  A=byte                      carry on error           (open write channel)
  CHRIN   --                          A=byte; carry on EOF     (open input channel)
  GETIN   --                          A=PETSCII key; carry     (the tap_key queue)
  SETLFS  A=L, X=drive, Y=device      A=0
  SETNAM  A=name hi, X=name lo, Y=type  A=0                   (name at (A,X))
  LOAD    A=verify 0/1, X=dst lo, Y=hi  A=0 ok/errno; X,Y=last byte; carry
  SAVE    A=start-word ptr, X=end lo, Y hi   A=0 ok/errno; carry
"""

_TRAPS = frozenset(
    (0xFFC0, 0xFFC3, 0xFFC6, 0xFFC9, 0xFFCC, 0xFFCF, 0xFFD2, 0xFFD5, 0xFFD8)
)
_EXT = {0: ".rel", 1: ".seq", 2: ".prg"}
_E_IO = 20
_E_NOCH = 14
_E_NOFILE = 40
_E_EOF = 24


def sys_addr(
    ram_or_image: Union["SidImage", bytes, bytearray], base: int = 0x0801
) -> int:
    """A single-line ``SYS`` stub at ``base`` returns the address it hands off to.

    ``SYS`` is BASIC token ``$64`` followed by the low then high byte of the target;
    this finds the first such token in ``base``..``base+256`` and returns it (LE) --
    enough for a one-``SYS``-line oracle stub. Accepts a ``SidImage`` or a buffer.

    Raises:
      KernalHostError: no ``SYS`` token in the region.
    """
    mem = (
        ram_or_image.mem if isinstance(ram_or_image, SidImage) else bytes(ram_or_image)
    )
    for i in range(base, min(len(mem) - 2, base + 256)):
        if mem[i] == 0x64:
            return mem[i + 1] | (mem[i + 2] << 8)
    raise KernalHostError(f"no SYS token in [${base:04x}, ${base + 256:04x})")


class KernalHost:
    """A C64 6502 whose KERNAL file/keyboard calls run on the host directory.

    ``kernal_rom``/``basic_rom`` bank real ROMs in when given or resolvable; offline
    they fall back to the ``irqhost`` KERNAL stub and zeros (the traps are Python).
    """

    def __init__(
        self,
        host_dir: Union[str, Path],
        *,
        ntsc: bool = False,
        kernal_rom: Optional[bytes] = None,
        basic_rom: Optional[bytes] = None,
    ):
        self.ntsc = ntsc
        self._host_dir = Path(host_dir)
        self._kernal_override = kernal_rom
        self._basic_override = basic_rom
        self._ram = bytearray(MEM_SIZE)
        self._ram[0] = 0x00
        self._ram[1] = 0x37  # power-on bank: KERNAL + BASIC in, CHAR out
        self.mem = _BankedMemory(self._ram, roms=self._roms())
        self.mpu = MPU(memory=self.mem)
        self._keys: Deque[int] = deque()
        self._lfn, self._drive, self._device = 1, 8, 8
        self._name, self._filetype = "", 1
        self._channels: Dict[int, dict] = {}
        self.touched: Set[int] = set()
        self.steps = 0
        self.mem.subscribe_to_write(range(MEM_SIZE), self._touch)

    def _roms(self) -> dict:
        """Resolve the KERNAL/BASIC ROM bytes (stub/zeros when offline)."""

        def _resolve(name: str, fallback: bytes) -> bytes:
            path = roms.resolve_rom(name)
            return Path(path).read_bytes() if path is not None else fallback

        kernal = self._kernal_override
        if kernal is None:
            kernal = _resolve("kernal", self._stub())
        basic = self._basic_override
        if basic is None:
            basic = _resolve("basic", bytes(0x2000))
        return {0xE000: bytes(kernal), 0xA000: bytes(basic)}

    @staticmethod
    def _stub() -> bytes:  # lazy: avoids the emu<->irqhost import cycle
        from .irqhost import kernal_stub

        return kernal_stub()

    def ram(self) -> bytearray:
        """The machine RAM a CPU write reaches (the CPU's whole address space)."""
        return self._ram

    def tap_key(self, petscii: int) -> "KernalHost":
        """Queue one PETSCII ``petscii`` byte for the next ``GETIN`` (chainable)."""
        self._keys.append(petscii & 0xFF)
        return self

    def screen(self) -> bytes:
        """The screen RAM, ``$0400``..``$07FF``."""
        return bytes(self._ram[0x0400:0x0800])

    def load_prg(self, prg: Union[bytes, bytearray, "SidImage"]) -> int:
        """Place a PRG (raw load+body or a ``SidImage``) and arm the CPU at it."""
        load, body = self._split(prg)
        self._ram[load : load + len(body)] = body
        self._reset_cpu(load)
        return load

    def run(
        self,
        until: Optional[Callable[["KernalHost"], bool]] = None,
        max_steps: int = 2_000_000,
    ) -> None:
        """Step (trapping KERNAL calls) until ``until`` is true or the cap is hit.

        Every RAM write is tracked in :attr:`touched`; raises
        :class:`~pysidtracker.errors.KernalHostError` on the step cap.
        """
        for _ in range(max_steps):
            if until is not None and until(self):
                return
            self.steps += 1
            self._step()
        raise KernalHostError(
            f"hit the {max_steps}-step cap before ``until`` returned True"
        )

    def run_to_unpacked(
        self, prg, entry: int = 0x0801, max_steps: int = 2_000_000
    ) -> SidImage:
        """Run a ``$entry`` packer until it jumps into RAM it wrote; return that.

        Returns the contiguous run of written pages from that entry as a bare
        ``SidImage`` placed at the entry; raises ``KernalHostError`` on the cap.
        """
        load, body = self._split(prg)
        ram = self._ram
        ram[load : load + len(body)] = body
        clean = self._clean_pages(load, len(body))
        self.touched.clear()
        self._reset_cpu(entry)
        for _ in range(max_steps):
            page = self.mpu.pc >> 12
            if page not in clean and page in self.touched:
                break
            self.steps += 1
            self._step()
        else:
            raise KernalHostError(
                f"depacker never entered a freshly-written region within {max_steps} steps"
            )
        entry_addr = self.mpu.pc & 0xFFFF
        top = entry_addr >> 12
        while (top + 1) in self.touched:
            top += 1
        data = bytes(ram[entry_addr : (top + 1) << 12])
        return SidImage._place(  # pylint: disable=protected-access
            data, entry_addr, None, b""
        )

    # --- internals ------------------------------------------------------------

    @staticmethod
    def _split(prg: Union[bytes, bytearray, "SidImage"]) -> "tuple[int, bytes]":
        if isinstance(prg, SidImage):
            return prg.load, bytes(prg.image)
        return parse_prg(bytes(prg))

    @staticmethod
    def _clean_pages(load: int, length: int) -> Set[int]:
        if not length:
            return {load >> 12}
        return set(range(load >> 12, (load + length - 1) >> 12 + 1))

    def _reset_cpu(self, pc: int) -> None:
        mpu = self.mpu
        mpu.a = mpu.x = mpu.y = 0
        mpu.sp = 0xFF
        mpu.p = 0
        mpu.pc = pc & 0xFFFF

    def _touch(
        self, _address: int, _value: int
    ) -> None:  # pylint: disable=unused-argument
        self.touched.add(_address >> 12)

    def _step(self) -> None:
        """Advance one instruction, answering a trapped KERNAL call if it is one."""
        mpu = self.mpu
        ram = self._ram
        pc = mpu.pc & 0xFFFF
        if ram[pc] == 0x20:
            vec = ram[(pc + 1) & 0xFFFF] | (ram[(pc + 2) & 0xFFFF] << 8)
            if vec in _TRAPS:
                self._dispatch(vec)
                mpu.pc = (pc + 3) & 0xFFFF
                return
        mpu.step()

    def _dispatch(self, vec: int) -> None:
        if vec == 0xFFC0:
            self._open()
        elif vec == 0xFFC3:
            self._close()
        elif vec == 0xFFC6:
            self._chout()
        elif vec == 0xFFC9:
            self._chrin()
        elif vec == 0xFFCC:
            self._getin()
        elif vec == 0xFFCF:
            self._setlfs()
        elif vec == 0xFFD2:
            self._setnam()
        elif vec == 0xFFD5:
            self._load()
        elif vec == 0xFFD8:
            self._save()

    def _ok(self) -> None:
        mpu = self.mpu
        mpu.a = 0
        mpu.p = mpu.p & ~MPU.CARRY

    def _err(self, code: int) -> None:
        mpu = self.mpu
        mpu.a = code
        mpu.p = mpu.p | MPU.CARRY

    def _getin(self) -> None:
        mpu = self.mpu
        if not self._keys:
            mpu.a = 0
            mpu.p = mpu.p | MPU.CARRY
            return
        mpu.a = self._keys.popleft()
        mpu.p = mpu.p & ~MPU.CARRY

    def _chrin(self) -> None:
        mpu = self.mpu
        ch = self._channels.get(self._lfn)
        if ch is None or ch["mode"] != "r":
            self._err(_E_NOCH)
            return
        data = ch["fh"].read(1)
        if not data:
            self._err(_E_EOF)
            return
        mpu.a = data[0]
        mpu.p = mpu.p & ~MPU.CARRY

    def _chout(self) -> None:
        mpu = self.mpu
        writers = [ch for ch in self._channels.values() if ch["mode"] == "w"]
        if not writers:
            self._err(_E_NOCH)
            return
        for ch in writers:
            ch["fh"].write(bytes([mpu.a]))
        mpu.a = 0
        mpu.p = mpu.p & ~MPU.CARRY

    def _open(self) -> None:
        try:
            path = self._path()
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                fh, mode = open(path, "rb"), "r"
            else:
                fh, mode = open(path, "wb"), "w"
            self._channels[self._lfn] = {"fh": fh, "mode": mode, "path": path}
        except OSError:
            self._err(_E_IO)
            return
        self._ok()

    def _close(self) -> None:
        mpu = self.mpu
        channels = list(self._channels) if mpu.a == 0xFF else [mpu.a]
        for lfn in channels:
            ch = self._channels.pop(lfn, None)
            if ch is not None:
                ch["fh"].close()
        self._ok()

    def _setlfs(self) -> None:
        mpu = self.mpu
        self._lfn, self._drive, self._device = mpu.a, mpu.x, mpu.y
        self._ok()

    def _setnam(self) -> None:
        mpu = self.mpu
        ram = self._ram
        addr = (mpu.a << 8) | mpu.x
        n = ram[addr]
        self._name = bytes(ram[addr + 1 : addr + 1 + n]).decode("latin-1")
        self._filetype = mpu.y
        self._ok()

    def _path(self) -> Path:
        name = self._name
        if "." not in name:
            name = name + _EXT.get(self._filetype, "")
        return self._host_dir / str(self._drive) / name

    def _save(self) -> None:
        mpu = self.mpu
        ram = self._ram
        start = (ram[mpu.a] | (ram[mpu.a + 1] << 8)) & 0xFFFF
        end = ((mpu.y << 8) | mpu.x) & 0xFFFF
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(ram[start:end]))
        self._ok()

    def _load(self) -> None:
        mpu = self.mpu
        ram = self._ram
        dst = ((mpu.y << 8) | mpu.x) & 0xFFFF
        path = self._path()
        if not path.exists():
            self._err(_E_NOFILE)
            return
        data = path.read_bytes()
        if mpu.a == 1:  # verify: compare only, leaving RAM untouched
            ok = bytes(ram[dst : dst + len(data)]) == data
            mpu.a = 0 if ok else 1
            mpu.p = (mpu.p & ~MPU.CARRY) if ok else (mpu.p | MPU.CARRY)
            return
        ram[dst : dst + len(data)] = data
        last = (dst + len(data) - 1) & 0xFFFF
        mpu.a = 0
        mpu.x = last & 0xFF
        mpu.y = last >> 8
        mpu.p = mpu.p & ~MPU.CARRY


__all__ = ["ABI", "KernalHost", "sys_addr"]
