package com.example.facercognitionapp.network

import com.example.facercognitionapp.model.FaceLoginRequest
import com.example.facercognitionapp.model.FaceLoginResponse
import com.example.facercognitionapp.model.FaceRecognizeResponse
import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

/** InsightFace server (SCRFD + AdaFace), see [FaceApiClient]. */
interface FaceApiService {

    /** POST /api/login  {"username","password"} -> {"token"} */
    @POST("api/login")
    suspend fun login(@Body request: FaceLoginRequest): Response<FaceLoginResponse>

    /**
     * POST /api/recognize
     * form-data: image (jpeg), source, punch ("true" = record attendance, with a cooldown)
     */
    @Multipart
    @POST("api/recognize")
    suspend fun recognize(
        @Part image: MultipartBody.Part,
        @Part("source") source: RequestBody,
        @Part("punch") punch: RequestBody
    ): Response<FaceRecognizeResponse>
}
