namespace CCTV_Backend_Project.Security
{
    /// <summary>The three roles in the roles table. WHAT each role may do is decided
    /// here and in the [Authorize(Roles = ...)] attribute on each controller action.</summary>
    public static class Roles
    {
        public const string SuperAdmin = "SUPER_ADMIN";
        public const string Admin = "ADMIN";
        public const string Viewer = "VIEWER";

        public const string AdminOrAbove = SuperAdmin + "," + Admin;
        public const string Anyone = SuperAdmin + "," + Admin + "," + Viewer;

        public static readonly string[] All = { SuperAdmin, Admin, Viewer };

        public static int Rank(string role) => role switch
        {
            SuperAdmin => 100,
            Admin => 50,
            Viewer => 10,
            _ => 0,
        };

        /// <summary>Rank rule: a login may only manage logins of a strictly lower rank;
        /// a SUPER_ADMIN may also manage other SUPER_ADMINs.</summary>
        public static bool CanManage(string actorRole, string targetRole)
            => actorRole == SuperAdmin || Rank(actorRole) > Rank(targetRole);
    }
}
