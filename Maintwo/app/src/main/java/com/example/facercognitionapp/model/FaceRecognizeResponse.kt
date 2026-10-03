package com.example.facercognitionapp.model

import com.google.gson.annotations.SerializedName

/**
 * Body of InsightFace POST /api/recognize.
 * [reason]: "identify_only" | "punched" | "cooldown" | "unknown" | "no_face"
 */
data class FaceRecognizeResponse(
    @SerializedName("recognized") val recognized: Boolean = false,
    @SerializedName("reason") val reason: String? = null,
    @SerializedName("message") val message: String? = null,
    @SerializedName("confidence") val confidence: Double? = null,
    @SerializedName("best_score") val bestScore: Double? = null,
    @SerializedName("face_width_px") val faceWidthPx: Int? = null,
    @SerializedName("low_confidence") val lowConfidence: Boolean? = null,
    @SerializedName("remaining_seconds") val remainingSeconds: Int? = null,
    @SerializedName("user") val user: FaceUser? = null
)

data class FaceUser(
    @SerializedName("id") val id: Int,
    @SerializedName("name") val name: String?,
    @SerializedName("employee_id") val employeeId: String?,
    /** Server path of the full photo the user registered with, e.g. /data/enrollment/3/..._original.jpg */
    @SerializedName("photo_url") val photoUrl: String? = null
)
