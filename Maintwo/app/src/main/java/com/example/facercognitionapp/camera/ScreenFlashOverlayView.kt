package com.example.facercognitionapp.camera

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.util.AttributeSet
import android.view.View

/**
 * A custom view that provides screen illumination (flash) in low-light conditions
 * by showing a solid white background covering the screen.
 */
class ScreenFlashOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    init {
        setBackgroundColor(Color.WHITE)
    }
}
