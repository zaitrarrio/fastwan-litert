# LiteRT's Kotlin API reaches its natives through JNI; keep it whole if minification is enabled.
-keep class com.google.ai.edge.litert.** { *; }
# kotlinx.serialization: keep the generated serializers of :core's manifest classes.
-keep,includedescriptorclasses class dev.fastwan.litert.core.**$$serializer { *; }
-keepclassmembers class dev.fastwan.litert.core.** { *** Companion; }
-keepclasseswithmembers class dev.fastwan.litert.core.** { kotlinx.serialization.KSerializer serializer(...); }
