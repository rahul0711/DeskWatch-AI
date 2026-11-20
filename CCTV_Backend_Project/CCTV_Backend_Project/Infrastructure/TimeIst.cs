using System.Globalization;

namespace CCTV_Backend_Project.Infrastructure
{
    /// <summary>"Day" in reports means an India (UTC+05:30) calendar day; the
    /// database stores UTC.</summary>
    public static class TimeIst
    {
        public static readonly TimeSpan Offset = new(5, 30, 0);

        public static DateOnly Today() => DateOnly.FromDateTime(DateTime.UtcNow + Offset);

        public static DateOnly ParseOrToday(string? value, string name = "date")
        {
            if (string.IsNullOrWhiteSpace(value)) return Today();
            if (DateOnly.TryParseExact(value, "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out var d)) return d;
            throw ApiException.BadRequest($"{name} must be YYYY-MM-DD");
        }

        /// <summary>[day 00:00 IST, next day 00:00 IST) as a UTC range.</summary>
        public static (DateTime FromUtc, DateTime ToUtc) DayToUtcRange(DateOnly day)
        {
            var start = day.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc) - Offset;
            return (start, start.AddDays(1));
        }

        /// <summary>A naive IST value from a view (*_ist column) with its +05:30 offset.</summary>
        public static DateTimeOffset AsIst(DateTime naiveIst)
            => new(DateTime.SpecifyKind(naiveIst, DateTimeKind.Unspecified), Offset);
    }
}
