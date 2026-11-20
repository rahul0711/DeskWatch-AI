using System.Data.Common;
using MySql.Data.MySqlClient;

namespace CCTV_Backend_Project.Infrastructure
{
    /// <summary>Opens MySQL connections from ConnectionStrings:WebPortalMYSQLConnection.
    /// MySql.Data pools connections, so open one per operation and dispose it.</summary>
    public class Db
    {
        private readonly string _connectionString;

        public Db(IConfiguration configuration)
        {
            _connectionString = configuration.GetConnectionString("WebPortalMYSQLConnection")
                ?? throw new InvalidOperationException("ConnectionStrings:WebPortalMYSQLConnection is missing in appsettings.json");
        }

        public async Task<MySqlConnection> OpenAsync(CancellationToken ct)
        {
            var conn = new MySqlConnection(_connectionString);
            await conn.OpenAsync(ct);
            return conn;
        }
    }

    public static class DbExtensions
    {
        public static MySqlCommand Command(this MySqlConnection conn, string sql, MySqlTransaction? tx = null)
            => new MySqlCommand(sql, conn, tx);

        public static MySqlCommand With(this MySqlCommand cmd, string name, object? value)
        {
            cmd.Parameters.AddWithValue(name, value ?? DBNull.Value);
            return cmd;
        }

        /// <summary>Every DATETIME column holds UTC without a zone.</summary>
        public static DateTime GetUtc(this DbDataReader r, string column)
            => DateTime.SpecifyKind(r.GetDateTime(r.GetOrdinal(column)), DateTimeKind.Utc);

        public static DateTime? GetUtcOrNull(this DbDataReader r, string column)
        {
            var i = r.GetOrdinal(column);
            return r.IsDBNull(i) ? null : DateTime.SpecifyKind(r.GetDateTime(i), DateTimeKind.Utc);
        }

        public static string? GetStringOrNull(this DbDataReader r, string column)
        {
            var i = r.GetOrdinal(column);
            return r.IsDBNull(i) ? null : r.GetString(i);
        }

        public static int? GetIntOrNull(this DbDataReader r, string column)
        {
            var i = r.GetOrdinal(column);
            return r.IsDBNull(i) ? null : Convert.ToInt32(r.GetValue(i));
        }

        public static int GetInt(this DbDataReader r, string column) => Convert.ToInt32(r.GetValue(r.GetOrdinal(column)));

        public static long GetLong(this DbDataReader r, string column) => Convert.ToInt64(r.GetValue(r.GetOrdinal(column)));

        public static double GetDouble(this DbDataReader r, string column)
        {
            var i = r.GetOrdinal(column);
            return r.IsDBNull(i) ? 0 : Convert.ToDouble(r.GetValue(i));
        }

        public static bool GetBool(this DbDataReader r, string column) => Convert.ToInt32(r.GetValue(r.GetOrdinal(column))) != 0;

        /// <summary>Builds "@p0, @p1, ..." for an IN (...) list and adds the parameters.</summary>
        public static string InList(this MySqlCommand cmd, string prefix, IReadOnlyList<string> values)
        {
            if (values.Count == 0) return "NULL";
            var names = new string[values.Count];
            for (var i = 0; i < values.Count; i++)
            {
                names[i] = $"@{prefix}{i}";
                cmd.Parameters.AddWithValue(names[i], values[i]);
            }
            return string.Join(", ", names);
        }
    }
}
