"""The VIC-II and CIA ports against documented 6526/6569 register behaviour."""

from pysidtracker.cia import Cia
from pysidtracker.emu import SidReads
from pysidtracker.eventsched import Event, Scheduler
from pysidtracker.vicii import Vic


def _cia():
    sched, lines = Scheduler(), []
    cia = Cia(sched, lambda state: lines.append((sched.time(0), state)), new=False)
    cia.reset()
    return sched, cia, lines


def _at(sched, cycle):
    """Run the chips to ``cycle`` and stand at its CPU (PHI2) slot."""
    sched.run(2 * cycle)
    sched.now = 2 * cycle + 1


def test_icr_read_reports_then_clears_the_flags():
    sched, cia, lines = _cia()
    _at(sched, 2)
    cia.write(0x04, 0x05)
    cia.write(0x05, 0x00)
    cia.write(0x0D, 0x81)
    cia.write(0x0E, 0x19)
    _at(sched, 40)
    assert lines == [(lines[0][0], True)]
    assert cia.read(0x0D) == 0x81
    _at(sched, 45)
    assert cia.read(0x0D) == 0x00
    assert lines[-1][1] is False


def test_timer_b_cascades_timer_a_underflows():
    sched, cia, _ = _cia()
    _at(sched, 2)
    for addr, value in ((4, 9), (5, 0), (6, 2), (7, 0), (0x0F, 0x51), (0x0E, 0x11)):
        cia.write(addr, value)
    seen = set()
    for cycle in range(3, 200):
        _at(sched, cycle)
        seen.add(cia.read(0x06))
    assert seen == {0, 1, 2}
    assert cia.read(0x0F) & 0x01 and cia.read(0x0E) & 0x01


def test_one_shot_stops_and_force_load_reloads():
    sched, cia, _ = _cia()
    _at(sched, 2)
    cia.write(0x04, 0x20)
    cia.write(0x05, 0x00)
    cia.write(0x0E, 0x19)
    _at(sched, 100)
    assert cia.read(0x0E) & 0x01 == 0 and cia.read(0x04) == 0x20
    cia.write(0x04, 0x10)
    cia.write(0x0E, 0x10)
    _at(sched, 104)
    assert cia.read(0x04) == 0x10


def test_timer_outputs_toggle_port_b():
    sched, cia, _ = _cia()
    _at(sched, 2)
    cia.write(0x04, 0x03)
    cia.write(0x05, 0x00)
    cia.write(0x0E, 0x17)
    levels = set()
    for cycle in range(3, 40):
        _at(sched, cycle)
        levels.add(cia.read(0x01) & 0x40)
    assert levels == {0x00, 0x40}
    cia.write(0x0F, 0x02)
    assert cia.read(0x01) & 0x80 == 0


def test_unmodelled_registers_read_back_writes():
    sched, cia, _ = _cia()
    _at(sched, 2)
    cia.write(0x08, 0x12)
    cia.write(0x02, 0xFF)
    cia.write(0x00, 0x0F)
    assert (cia.read(0x08), cia.read(0x00)) == (0x12, 0x0F)


def test_vic_register_reads():
    sched = Scheduler()
    vic = Vic(sched, False, lambda _s: None, lambda _s: None)
    _at(sched, 70)
    vic.write(0x1A, 0x01)
    vic.write(0x20, 0x0E)
    assert (vic.read(0x1A), vic.read(0x20), vic.read(0x30)) == (0xF1, 0xFE, 0xFF)
    assert (vic.read(0x12), vic.read(0x13), vic.read(0x15)) == (0, 0, 0)


def test_sprite_crunch_mixes_the_data_counters():
    sched = Scheduler()
    vic = Vic(sched, False, lambda _s: None, lambda _s: None)
    sprites = vic.sprites
    sprites.exp_flop, sprites.mc, sprites.mc_base = 0x00, [0x3F] * 8, [0x00] * 8
    vic.line_cycle = 14
    sprites.line_crunch(0x00, 14)
    assert sprites.mc[0] == 0x15 and sprites.exp_flop == 0xFF


def test_scheduler_orders_same_slot_events_by_scheduling():
    sched, fired = Scheduler(), []
    first, second = Event(lambda: fired.append(1)), Event(lambda: fired.append(2))
    sched.schedule(first, 1, 0)
    sched.schedule(second, 1, 0)
    sched.cancel(first)
    sched.schedule(first, 1, 0)
    sched.run(10)
    assert fired == [2, 1] and sched.time(1) == 1


def test_sid_reads_flush_queued_writes():
    chip = SidReads("8580")
    chip.write(10, 0x12, 0x81)
    assert chip.pending() == 1
    chip.flush(20)
    assert chip.pending() == 0
    assert chip.read(30, 0x19) == chip.read(30, 0x19)
