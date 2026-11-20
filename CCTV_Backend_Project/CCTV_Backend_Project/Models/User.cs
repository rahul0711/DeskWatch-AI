namespace CCTV_Backend_Project.Models
{
    /// <summary>Tracked person (table: users). Written by the Python attendance server only.</summary>
    public class User
    {
        public int Id { get; set; }

        /// <summary>Unique employee code entered at registration.</summary>
        public string EmployeeId { get; set; } = string.Empty;

        public string Name { get; set; } = string.Empty;

        /// <summary>UTC.</summary>
        public DateTime CreatedAt { get; set; }
    }
}
