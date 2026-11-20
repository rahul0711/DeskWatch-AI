namespace CCTV_Backend_Project.Models
{
    /// <summary>Table: cameras.</summary>
    public class Camera
    {
        /// <summary>Primary key. Slug used everywhere else: camera-01 | inside | corridor.</summary>
        public string CameraId { get; set; } = string.Empty;

        /// <summary>Camera 1, 2, 3 as shown in the UI (unique).</summary>
        public int CameraNo { get; set; }

        public string Name { get; set; } = string.Empty;

        public string? IpAddress { get; set; }

        public string? Brand { get; set; }

        public bool IsActive { get; set; } = true;

        /// <summary>Face detection + recognition (attendance, names, restricted-area alerts).</summary>
        public bool FaceEnabled { get; set; } = true;

        /// <summary>Body movement: posture, movement, phone / monitor / computer use.</summary>
        public bool ActivityEnabled { get; set; }
    }
}
