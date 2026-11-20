namespace CCTV_Backend_Project.Models
{
    /// <summary>Camera granted to a login (table: app_user_cameras). Primary key is (AppUserId, CameraId).</summary>
    public class AppUserCamera
    {
        /// <summary>FK to app_users.id (ON DELETE CASCADE).</summary>
        public int AppUserId { get; set; }

        /// <summary>FK to cameras.camera_id (ON DELETE/UPDATE CASCADE).</summary>
        public string CameraId { get; set; } = string.Empty;

        /// <summary>app_users.id of the granter.</summary>
        public int? GrantedBy { get; set; }

        /// <summary>UTC.</summary>
        public DateTime CreatedAt { get; set; }
    }
}
