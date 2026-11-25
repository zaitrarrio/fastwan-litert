package dev.fastwan.litert.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.fastwan.litert.app.ui.FastWanTheme
import dev.fastwan.litert.app.ui.GenerateScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            FastWanTheme {
                val vm: GenerateViewModel = viewModel()
                GenerateScreen(vm)
            }
        }
    }
}
