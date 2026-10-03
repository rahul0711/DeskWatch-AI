package com.example.facercognitionapp.camera

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.google.mlkit.vision.face.Face

class FaceOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val boxPaint = Paint().apply {
        color = Color.argb(180, 0, 255, 0)
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
        pathEffect = DashPathEffect(floatArrayOf(16f, 10f), 0f)
    }

    private var faces: List<Face> = emptyList()

    // Camera image size
    private var imageWidth = 0
    private var imageHeight = 0

    // Front camera mirror
    private var isFrontCamera = true

    fun setFaces(
        faces: List<Face>,
        imageWidth: Int,
        imageHeight: Int,
        isFrontCamera: Boolean
    ) {
        this.faces = faces
        this.imageWidth = imageWidth
        this.imageHeight = imageHeight
        this.isFrontCamera = isFrontCamera
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (faces.isEmpty() || imageWidth == 0 || imageHeight == 0) return

        val scaleX = width.toFloat() / imageWidth
        val scaleY = height.toFloat() / imageHeight

        faces.forEach { face ->

            val rect = RectF(face.boundingBox)

            // Scale to view
            rect.left *= scaleX
            rect.right *= scaleX
            rect.top *= scaleY
            rect.bottom *= scaleY

            // Mirror for front camera
            if (isFrontCamera) {
                val left = rect.left
                rect.left = width - rect.right
                rect.right = width - left
            }

            canvas.drawRect(rect, boxPaint)
        }
    }
}