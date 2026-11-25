package dev.fastwan.litert.core

import java.io.File

/** The tiny export's manifest and expected tensors, copied from `fixtures/tiny`. */
object Fixture {
    fun text(name: String): String =
        Fixture::class.java.getResourceAsStream("/tiny/$name")!!.use { String(it.readBytes(), Charsets.UTF_8) }

    fun bytes(name: String): ByteArray = Fixture::class.java.getResourceAsStream("/tiny/$name")!!.use { it.readBytes() }

    fun manifest(dir: File? = null): Manifest = Manifest.parse(text("manifest.json"), dir)

    fun expected(): SafeTensors = SafeTensors.read(bytes("expected.safetensors"))
}
