"""Tests for the new register-index constants and ADSR packers."""

import pytest

from pysidtracker import SID_VOICES, attack_decay, sustain_release
from pysidtracker import registers as reg


@pytest.mark.strain_exempt(reason="exercises code outside the measured package")
def test_voice_constants():
    assert SID_VOICES == 3
    assert (reg.FREQ_LO, reg.FREQ_HI, reg.PW_LO, reg.PW_HI) == (0, 1, 2, 3)
    assert (reg.CTRL, reg.AD, reg.SR) == (4, 5, 6)
    assert (reg.FC_LO, reg.FC_HI, reg.RES_FILT, reg.MODE_VOL) == (
        0x15,
        0x16,
        0x17,
        0x18,
    )


@pytest.mark.parametrize("bad", [-1, 16, 255])
def test_adsr_range_checked(bad):
    with pytest.raises(ValueError):
        attack_decay(bad, 0)
    with pytest.raises(ValueError):
        sustain_release(0, bad)
