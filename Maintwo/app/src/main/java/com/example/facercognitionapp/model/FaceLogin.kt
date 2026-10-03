package com.example.facercognitionapp.model

import com.google.gson.annotations.SerializedName

data class FaceLoginRequest(
    @SerializedName("username") val username: String,
    @SerializedName("password") val password: String
)

data class FaceLoginResponse(
    @SerializedName("token") val token: String?
)
