package com.example.facercognitionapp

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.os.Build
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.View
import android.view.WindowManager
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.facercognitionapp.camera.CameraHelper
import com.example.facercognitionapp.databinding.ActivityMainBinding
import com.example.facercognitionapp.model.FaceRecognizeResponse
import com.example.facercognitionapp.network.FaceApiClient
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Attendance kiosk: full-screen front camera. When a face is seen, the photo goes to
 * InsightFace /api/recognize (which records the attendance) and the person is greeted
 * by name. The next scan starts once that person has left the frame.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraHelper: CameraHelper
    private var tts: TextToSpeech? = null
    private var ttsReady = false

    private var waitingForFaceToLeave = false
    private var resumeJob: Job? = null
    private var lastGreetedUserId: Int? = null
    private var lastGreetedAt = 0L
    private var resultGeneration = 0

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else showStatus(getString(R.string.status_camera_denied))
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        if (FaceApiClient.token == null && FaceApiClient.loadToken(this) == null) {
            goToLogin()
            return
        }

        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale.US
                ttsReady = true
            }
        }

        cameraHelper = CameraHelper(
            context = this,
            lifecycleOwner = this,
            onFaceDetected = ::onFaceCaptured,
            onNoFace = ::onNoFace,
            onLowLightChanged = ::setLowLight
        )

        binding.btnLogout.setOnClickListener { confirmLogout() }
        binding.resultPhoto.clipToOutline = true

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        cameraHelper.startCamera(binding.previewView)
        showStatus(getString(R.string.status_look_at_camera))
    }

    // ---------------- recognition ----------------

    private fun onFaceCaptured(photo: File) {
        cameraHelper.lockForApi()
        showStatus(getString(R.string.status_recognizing))
        lifecycleScope.launch { recognize(photo) }
    }

    private suspend fun recognize(photo: File) {
        val result: FaceRecognizeResponse = try {
            val textType = "text/plain".toMediaTypeOrNull()
            val response = FaceApiClient.api.recognize(
                image = MultipartBody.Part.createFormData(
                    "image", photo.name, photo.asRequestBody("image/jpeg".toMediaTypeOrNull())
                ),
                source = "Mobile - ${Build.MODEL}".toRequestBody(textType),
                punch = "true".toRequestBody(textType)
            )
            if (response.code() == 401) {
                FaceApiClient.clearToken(this)
                goToLogin()
                return
            }
            val body = response.body()
            if (!response.isSuccessful || body == null) {
                Log.w(TAG, "recognize HTTP ${response.code()}: ${response.errorBody()?.string()}")
                showResult(false, getString(R.string.result_error_title), getString(R.string.result_error_http, response.code()))
                resumeScanning(ERROR_DISPLAY_MS, requireFaceToLeave = false)
                return
            }
            body
        } catch (e: Exception) {
            Log.e(TAG, "recognize failed", e)
            showResult(false, getString(R.string.result_error_title), getString(R.string.error_server_unreachable))
            resumeScanning(ERROR_DISPLAY_MS, requireFaceToLeave = false)
            return
        } finally {
            photo.delete()
        }

        Log.i(
            TAG,
            "reason=${result.reason} user=${result.user?.name}/${result.user?.employeeId} " +
                "confidence=${result.confidence} best_score=${result.bestScore} face=${result.faceWidthPx}px"
        )

        when (result.reason) {
            "punched", "cooldown", "identify_only" -> greet(result)
            "no_face" -> {
                showStatus(getString(R.string.status_no_face))
                resumeScanning(NO_FACE_RETRY_MS, requireFaceToLeave = false)
            }
            else -> {
                showResult(false, getString(R.string.result_not_recognised), getString(R.string.result_not_recognised_hint))
                speak(getString(R.string.result_not_recognised))
                resumeScanning(ERROR_DISPLAY_MS, requireFaceToLeave = false)
            }
        }
    }

    private fun greet(result: FaceRecognizeResponse) {
        val user = result.user
        val name = user?.name?.takeIf { it.isNotBlank() } ?: getString(R.string.result_default_name)
        val now = System.currentTimeMillis()

        // Same person still standing in front after the fallback timeout: don't greet again.
        val repeat = user != null && user.id == lastGreetedUserId && now - lastGreetedAt < REPEAT_GREETING_MS
        lastGreetedUserId = user?.id
        lastGreetedAt = now
        if (repeat) {
            resumeScanning(0, requireFaceToLeave = true)
            return
        }

        val subtitle = if (result.reason == "cooldown") {
            val minutes = ((result.remainingSeconds ?: 0) + 59) / 60
            getString(R.string.result_already_marked, maxOf(1, minutes))
        } else {
            getString(R.string.result_marked_at, SimpleDateFormat("hh:mm a", Locale.getDefault()).format(Date()))
        }
        showResult(true, getString(R.string.result_welcome, name), subtitle)
        speak(getString(R.string.speech_welcome, name))
        user?.photoUrl?.let { showRegisteredPhoto(it) }
        resumeScanning(WELCOME_DISPLAY_MS, requireFaceToLeave = true)
    }

    /** Swaps the ✓ icon for the user's registration photo once it has downloaded. */
    private fun showRegisteredPhoto(path: String) {
        val generation = resultGeneration
        lifecycleScope.launch {
            val photo = FaceApiClient.loadPhoto(path) ?: return@launch
            if (generation != resultGeneration || binding.resultCard.visibility != View.VISIBLE) return@launch
            binding.resultPhoto.setImageBitmap(photo)
            binding.resultPhoto.visibility = View.VISIBLE
            binding.resultIconContainer.visibility = View.GONE
        }
    }

    /**
     * Hides the result after [delayMs], then re-enables capture. With [requireFaceToLeave]
     * capture resumes when the frame is empty (see [onNoFace]) or after a fallback timeout.
     */
    private fun resumeScanning(delayMs: Long, requireFaceToLeave: Boolean) {
        resumeJob?.cancel()
        resumeJob = lifecycleScope.launch {
            delay(delayMs)
            binding.resultCard.visibility = View.GONE
            if (requireFaceToLeave) {
                waitingForFaceToLeave = true
                showStatus(getString(R.string.status_next_person))
                delay(FACE_LEAVE_TIMEOUT_MS)
                waitingForFaceToLeave = false
            }
            unlockCamera()
        }
    }

    private fun onNoFace() {
        if (waitingForFaceToLeave) {
            waitingForFaceToLeave = false
            resumeJob?.cancel()
            unlockCamera()
        }
    }

    private fun unlockCamera() {
        setLowLight(false)
        cameraHelper.unlockAfterApi()
        showStatus(getString(R.string.status_look_at_camera))
    }

    // ---------------- UI ----------------

    private fun showResult(success: Boolean, title: String, subtitle: String) {
        resultGeneration++
        binding.resultPhoto.visibility = View.GONE
        binding.resultPhoto.setImageDrawable(null)
        binding.resultIconContainer.visibility = View.VISIBLE
        binding.resultTitle.text = title
        binding.resultSubtitle.text = subtitle
        binding.resultSubtitle.visibility = if (subtitle.isBlank()) View.GONE else View.VISIBLE
        binding.resultIcon.text = if (success) "✓" else "✕"
        binding.resultIconBg.backgroundTintList =
            ColorStateList.valueOf(if (success) COLOR_SUCCESS else COLOR_ERROR)
        binding.resultCard.visibility = View.VISIBLE
    }

    private fun showStatus(text: String) {
        binding.statusText.text = text
    }

    private fun speak(text: String) {
        if (ttsReady) tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "result")
    }

    private fun setLowLight(isDark: Boolean) {
        binding.screenFlashOverlay.visibility = if (isDark) View.VISIBLE else View.GONE
        window.attributes = window.attributes.apply {
            screenBrightness = if (isDark) {
                WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_FULL
            } else {
                WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE
            }
        }
    }

    private fun confirmLogout() {
        AlertDialog.Builder(this)
            .setMessage(R.string.logout_confirm)
            .setPositiveButton(R.string.action_logout) { _, _ ->
                FaceApiClient.clearToken(this)
                goToLogin()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun goToLogin() {
        startActivity(Intent(this, LoginActivity::class.java))
        finish()
    }

    override fun onDestroy() {
        super.onDestroy()
        resumeJob?.cancel()
        tts?.shutdown()
        if (::cameraHelper.isInitialized) cameraHelper.stopCamera()
    }

    companion object {
        private const val TAG = "FaceKiosk"
        private const val WELCOME_DISPLAY_MS = 10_000L
        private const val ERROR_DISPLAY_MS = 2500L
        private const val NO_FACE_RETRY_MS = 800L
        private const val FACE_LEAVE_TIMEOUT_MS = 10_000L
        private const val REPEAT_GREETING_MS = 30_000L
        private val COLOR_SUCCESS = 0xFF22C55E.toInt()
        private val COLOR_ERROR = 0xFFEF5350.toInt()
    }
}
