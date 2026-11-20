namespace CCTV_Backend_Project.Models
{
    /// <summary>Table: activity_types (lookup, seeded).</summary>
    public class ActivityType
    {
        /// <summary>Primary key, e.g. SITTING, PHONE_INTERACTION.</summary>
        public string Code { get; set; } = string.Empty;

        public string DisplayName { get; set; } = string.Empty;

        /// <summary>posture | movement | interaction | presence.</summary>
        public string Category { get; set; } = string.Empty;

        public int SortOrder { get; set; }
    }
}
