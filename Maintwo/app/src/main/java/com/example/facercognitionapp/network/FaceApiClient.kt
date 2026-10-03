package com.example.facercognitionapp.network

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.util.LruCache
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.ByteArrayInputStream
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

object FaceApiClient {

    private const val BASE_URL = "http://120.138.7.130:9005/"

    private const val PREFS = "auth"
    private const val KEY_TOKEN = "face_token"

    /** Token from /api/login; attached to every request once set. */
    @Volatile
    var token: String? = null
        private set

    fun loadToken(context: Context): String? {
        token = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(KEY_TOKEN, null)
        return token
    }

    fun saveToken(context: Context, value: String) {
        token = value
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(KEY_TOKEN, value).apply()
    }

    fun clearToken(context: Context) {
        token = null
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().remove(KEY_TOKEN).apply()
    }

    private val client: OkHttpClient by lazy {
        OkHttpClient.Builder()
            .addInterceptor { chain ->
                val builder = chain.request().newBuilder().header("Accept", "application/json")
                token?.let { builder.header("Authorization", "Bearer $it") }
                chain.proceed(builder.build())
            }
            .addInterceptor(HttpLoggingInterceptor().apply { level = HttpLoggingInterceptor.Level.BASIC })
            .connectTimeout(15, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .build()
    }

    private val photoCache = LruCache<String, Bitmap>(20)

    /**
     * Downloads a server image path (e.g. a user's registration photo), downscaled so its
     * longest side is about [maxSidePx] and rotated per EXIF. Cached; null on any failure.
     */
    suspend fun loadPhoto(path: String, maxSidePx: Int = 1024): Bitmap? {
        photoCache.get(path)?.let { return it }
        return withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder().url(BASE_URL.trimEnd('/') + path).build()
                val bytes = client.newCall(request).execute().use { resp ->
                    if (!resp.isSuccessful) return@withContext null
                    resp.body?.bytes() ?: return@withContext null
                }
                val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
                var sample = 1
                while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxSidePx) sample *= 2
                val decoded = BitmapFactory.decodeByteArray(
                    bytes, 0, bytes.size, BitmapFactory.Options().apply { inSampleSize = sample }
                ) ?: return@withContext null
                val rotated = applyExifRotation(bytes, decoded)
                photoCache.put(path, rotated)
                rotated
            } catch (e: Exception) {
                null
            }
        }
    }

    private fun applyExifRotation(bytes: ByteArray, bitmap: Bitmap): Bitmap {
        val degrees = when (
            ExifInterface(ByteArrayInputStream(bytes))
                .getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        ) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90f
            ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> return bitmap
        }
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    val api: FaceApiService by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(FaceApiService::class.java)
    }
}
