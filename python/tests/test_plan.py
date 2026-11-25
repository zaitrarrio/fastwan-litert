from fastwan_litert import plan
from fastwan_litert.config import TaeConfig, WanConfig
from fastwan_litert.geometry import FASTWAN_480P, MOBILE_480x272


def test_480p_f16_fits_16gb_with_two_resident_blocks():
    p = plan.plan(WanConfig.fastwan_1_3b(), TaeConfig.taew2_1(), FASTWAN_480P, "f16", 1024, 256, 3, 16 << 30)
    assert p.fits and p.tokens == 32760, str(p)
    assert p.attention_scratch == 12 * 256 * 32760 * 4


def test_phone_tier_int4_fits_2gb():
    p = plan.plan(WanConfig.fastwan_1_3b(), TaeConfig.taew2_1(), MOBILE_480x272, "int4", 1024, 256, 3, 2 << 30)
    assert p.fits, str(p)
    assert p.weights_total < 1.0 * (1 << 30)


def test_480p_does_not_fit_a_phone():
    p = plan.plan(WanConfig.fastwan_1_3b(), TaeConfig.taew2_1(), FASTWAN_480P, "int4", 1024, 256, 3, 2 << 30)
    assert not p.fits
