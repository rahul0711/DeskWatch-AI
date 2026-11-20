namespace CCTV_Backend_Project.Dtos
{
    // JSON is snake_case (Program.cs), matching what the frontend already reads
    // from the Python servers: FullName <-> "full_name".

    // ---- Auth ------------------------------------------------------------------
    public record LoginRequest(string Username, string Password);

    public record LoginResponse(string Token, DateTime ExpiresAt, MeResponse User);

    public record ChangePasswordRequest(string CurrentPassword, string NewPassword);

    public record CameraRef(string CameraId, int CameraNo, string Name);

    public record MeResponse(
        int Id, string Username, string FullName, string? Email, string Role, string RoleName,
        bool MustChangePassword, bool AllCameras, IReadOnlyList<CameraRef> Cameras);

    // ---- Logins (dashboard accounts) ---------------------------------------------
    public record LoginDto(
        int Id, string Username, string FullName, string? Email, string Role, bool IsActive,
        bool AllCameras, IReadOnlyList<string> CameraIds, bool MustChangePassword,
        DateTime? LockedUntil, DateTime? LastLoginAt, int? CreatedBy, DateTime CreatedAt, DateTime UpdatedAt);

    public record CreateLoginRequest(
        string Username, string FullName, string? Email, string Role, bool AllCameras,
        List<string>? CameraIds, string Password);

    /// <summary>Null = leave unchanged.</summary>
    public record UpdateLoginRequest(string? FullName, string? Email, string? Role, bool? IsActive, bool? AllCameras);

    public record SetLoginCamerasRequest(List<string> CameraIds);

    public record ResetPasswordRequest(string Password);

    public record RoleDto(string Code, string Name, int RankLevel, string Description);

    // ---- Cameras -------------------------------------------------------------------
    public record CameraDto(string CameraId, int CameraNo, string Name, string? IpAddress, string? Brand, bool IsActive,
        bool FaceEnabled, bool ActivityEnabled);

    public record UpdateCameraRequest(string? Name, bool? IsActive);

    /// <summary>Which analyses a camera runs; null leaves that switch as it is.</summary>
    public record UpdateCameraFeaturesRequest(bool? FaceEnabled, bool? ActivityEnabled);

    // ---- Tracked people ------------------------------------------------------------
    public record FaceDto(int Id, string ThumbnailUrl, string? OriginalUrl, int FaceWidthPx, double DetScore, DateTime CreatedAt);

    /// <summary>Same shape as the Python GET /api/users, so the Register page reads it unchanged.</summary>
    public record PersonDto(int Id, string EmployeeId, string Name, DateTime CreatedAt, List<string> BlockedCameras, List<FaceDto> Faces);

    public record CameraAccessRequest(List<string> BlockedCameras);

    public record CameraAccessResponse(int UserId, List<string> BlockedCameras);

    public record PersonCameraAccessDto(string CameraId, string CameraName, string Access, DateTime? BlockedSince);

    // ---- Attendance ----------------------------------------------------------------
    /// <summary>Same shape as the Python GET /api/attendance.</summary>
    public record AttendanceDto(
        int Id, int UserId, string Name, string EmployeeId, string CameraName, DateTime Timestamp,
        double Confidence, int FaceWidthPx, bool LowConfidence, string ThumbnailUrl);

    public record AttendanceDailyDto(
        int UserId, string EmployeeId, string Name, string? PhotoUrl, bool Present,
        DateTimeOffset? FirstIn, DateTimeOffset? LastSeen, int Punches, List<string> Cameras);

    // ---- Activity ------------------------------------------------------------------
    public record ActivityDailyDto(
        string? EmployeeId, string PersonName,
        double SittingSeconds, double StandingSeconds, double WalkingSeconds, double LowMovementSeconds,
        double PhoneSeconds, double MonitorSeconds, double ComputerSeconds, double PresentSeconds,
        int PhoneSessions, DateTimeOffset? FirstSeenIst, DateTimeOffset? LastSeenIst);

    public record ActivitySessionDto(
        long Id, string? EmployeeId, string PersonName, string CameraId, string? CameraName,
        string ActivityType, string? ActivityName, DateTime StartTime, DateTime EndTime,
        double DurationSeconds, double Confidence);

    public record ActivityTypeDto(string Code, string DisplayName, string Category, int SortOrder);

    // ---- Audit ---------------------------------------------------------------------
    public record AuditDto(
        long Id, DateTime At, int? AppUserId, string? Username, string Action, string? TargetType,
        string? TargetId, string? CameraId, string? Details, string? IpAddress);

    public record Paged<T>(IReadOnlyList<T> Items, int Page, int PageSize, long Total);
}
