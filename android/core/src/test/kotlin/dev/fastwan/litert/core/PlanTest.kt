package dev.fastwan.litert.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** `python/tests/test_plan.py`. */
class PlanTest {
    @Test
    fun p480F16Fits16GbWithTwoResidentBlocks() {
        val p = Plan.compute(WanConfig.fastwan13b(), TaeConfig.taew21(), Geometry.FASTWAN_480P, "f16", 1024, 256, 3, 16L shl 30)
        assertTrue(p.fits && p.tokens == 32760, p.toString())
        assertEquals(12L * 256 * 32760 * 4, p.attentionScratch)
    }

    @Test
    fun phoneTierInt4Fits2Gb() {
        val p = Plan.compute(WanConfig.fastwan13b(), TaeConfig.taew21(), Geometry.MOBILE_480x272, "int4", 1024, 256, 3, 2L shl 30)
        assertTrue(p.fits, p.toString())
        assertTrue(p.weightsTotal < (1L shl 30), p.toString())
    }

    @Test
    fun p480DoesNotFitAPhone() {
        val p = Plan.compute(WanConfig.fastwan13b(), TaeConfig.taew21(), Geometry.FASTWAN_480P, "int4", 1024, 256, 3, 2L shl 30)
        assertFalse(p.fits)
        assertTrue(p.toString().contains("fits: false"))
    }
}
