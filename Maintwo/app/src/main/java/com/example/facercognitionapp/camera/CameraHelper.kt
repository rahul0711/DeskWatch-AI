package com.example.facercognitionapp.camera

import android.content.Context
import android.util.Log
import android.util.Size
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.io.File
import java.util.concurrent.Executors

/**
 * Camera capture + face detection logic only.
 * UI shell (sidebar + universal header) lives in [com.example.facercognitionapp.MainActivity]
 * via [com.example.facercognitionapp.ui.core.BaseDrawerContentActivity].
 */
class CameraHelper(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val onFaceDetected: (File) -> Unit,
    private val onNoFace: () -> Unit,
    private val onLowLightChanged: (Boolean) -> Unit = {}
) {

    private val executor = Executors.newSingleThreadExecutor()
    private lateinit var imageCapture: ImageCapture

    private var isCapturing = false
    /** Blocks new photos while Recognize API call is in progress. */
    private var apiLocked = false

    private var cameraStartTime = 0L
    private val warmupMs = 500L

    private var noFaceFrameCount = 0
    private val noFaceThreshold = 5

    private var lastAnalysisTime = 0L
    private val analysisIntervalMs = 120L

    private var isLowLightActive = false

    // ✅ Optimized ML Kit detector
    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setMinFaceSize(0.15f)
            .enableTracking()
            .build()
    )

    fun startCamera(previewView: PreviewView) {

        cameraStartTime = System.currentTimeMillis()

        val providerFuture = ProcessCameraProvider.getInstance(context)

        providerFuture.addListener({

            val cameraProvider = providerFuture.get()

            val preview = Preview.Builder().build().apply {
                setSurfaceProvider(previewView.surfaceProvider)
            }

            imageCapture = ImageCapture.Builder()
                .setTargetResolution(Size(720, 720))
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(720, 720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            analysis.setAnalyzer(executor, ::analyze)

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_FRONT_CAMERA,
                preview,
                analysis,
                imageCapture
            )

        }, ContextCompat.getMainExecutor(context))
    }

    // ================= FACE ANALYSIS =================

    @androidx.annotation.OptIn(ExperimentalGetImage::class)
    @OptIn(ExperimentalGetImage::class)
    private fun analyze(imageProxy: ImageProxy) {

        // Warmup
        if (System.currentTimeMillis() - cameraStartTime < warmupMs) {
            imageProxy.close()
            return
        }

        // Throttle (keeps CPU stable)
        val now = System.currentTimeMillis()
        if (now - lastAnalysisTime < analysisIntervalMs) {
            imageProxy.close()
            return
        }
        lastAnalysisTime = now

        // Calculate average luminance (brightness) of the frame
        try {
            val yPlane = imageProxy.planes[0]
            val buffer = yPlane.buffer
            val start = buffer.position()
            val remaining = buffer.remaining()
            
            var sum = 0L
            var count = 0
            val step = maxOf(50, remaining / 1000)
            
            for (i in 0 until remaining step step) {
                sum += buffer.get(start + i).toInt() and 0xFF
                count++
            }
            
            val avgLuminance = if (count > 0) sum.toDouble() / count else 0.0
            val isDark = if (isLowLightActive) {
                avgLuminance < 60.0
            } else {
                avgLuminance < 45.0
            }
            
            if (!apiLocked && !isCapturing && isDark != isLowLightActive) {
                isLowLightActive = isDark
                ContextCompat.getMainExecutor(context).execute {
                    onLowLightChanged(isLowLightActive)
                }
            }
        } catch (e: Exception) {
            Log.e("CameraHelper", "Luminance calculation failed", e)
        }

        val mediaImage = imageProxy.image ?: run {
            imageProxy.close()
            return
        }

        val image = InputImage.fromMediaImage(
            mediaImage,
            imageProxy.imageInfo.rotationDegrees
        )

        detector.process(image)
            .addOnSuccessListener { faces ->

                if (faces.isEmpty()) {
                    noFaceFrameCount++
                    if (noFaceFrameCount >= noFaceThreshold) {
                        onNoFace()
                    }
                    imageProxy.close()
                    return@addOnSuccessListener
                }

                noFaceFrameCount = 0

                // ✅ Face size filter (important for queue accuracy)
                val face = faces[0]
                if (face.boundingBox.width() < 200) {
                    imageProxy.close()
                    return@addOnSuccessListener
                }

                // Only capture when not busy and API is not in flight
                if (!isCapturing && !apiLocked) {
                    isCapturing = true
                    capture()
                }

                imageProxy.close()
            }
            .addOnFailureListener {
                imageProxy.close()
            }
    }

    // ================= CAPTURE =================

    private fun capture() {

        val photoFile = File(
            context.cacheDir,
            "face_${System.currentTimeMillis()}.jpg"
        )

        val outputOptions =
            ImageCapture.OutputFileOptions.Builder(photoFile).build()

        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageSavedCallback {

                override fun onImageSaved(
                    output: ImageCapture.OutputFileResults
                ) {
                    // 🔥 DO NOT unlock here
                    // MainActivity will reset after API response
                    onFaceDetected(photoFile)
                }

                override fun onError(exc: ImageCaptureException) {
                    isCapturing = false
                }
            }
        )
    }

    // ================= CONTROL =================

    /** Call before starting Recognize API — prevents duplicate captures. */
    fun lockForApi() {
        apiLocked = true
    }

    /** Call after Recognize API completes (success or failure). */
    fun unlockAfterApi() {
        apiLocked = false
        isCapturing = false
        isLowLightActive = false
    }

    @Deprecated("Use unlockAfterApi()", ReplaceWith("unlockAfterApi()"))
    fun resetCapture() {
        unlockAfterApi()
    }

    // ================= CLEANUP =================

    fun stopCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.get().unbindAll()
        executor.shutdown()
    }
}