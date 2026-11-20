namespace CCTV_Backend_Project.Models
{
    /// <summary>Table: roles (lookup, seeded: SUPER_ADMIN, ADMIN, VIEWER).</summary>
    public class Role
    {
        /// <summary>Primary key.</summary>
        public string Code { get; set; } = string.Empty;

        public string Name { get; set; } = string.Empty;

        /// <summary>Higher = more senior role (unique).</summary>
        public int RankLevel { get; set; }

        public string Description { get; set; } = string.Empty;
    }
}
