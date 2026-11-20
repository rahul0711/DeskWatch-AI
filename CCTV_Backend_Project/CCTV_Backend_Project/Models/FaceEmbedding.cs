namespace CCTV_Backend_Project.Models
{
    /// <summary>Table: face_embeddings. Written by the Python attendance server only.</summary>
    public class FaceEmbedding
    {
        public int Id { get; set; }

        /// <summary>FK to users.id (ON DELETE CASCADE).</summary>
        public int UserId { get; set; }

        /// <summary>512 x float32, little-endian.</summary>
        public byte[] Embedding { get; set; } = Array.Empty<byte>();

        /// <summary>File on disk under data/enrollment/.</summary>
        public string ThumbnailPath { get; set; } = string.Empty;

        public string? OriginalPath { get; set; }

        public int FaceWidthPx { get; set; }

        public double DetScore { get; set; }

        /// <summary>UTC.</summary>
        public DateTime CreatedAt { get; set; }
    }
}
