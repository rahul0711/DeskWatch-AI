using System.Security.Claims;
using CCTV_Backend_Project.Infrastructure;

namespace CCTV_Backend_Project.Security
{
    /// <summary>The logged-in login for this request, read from the JWT claims.</summary>
    public sealed record CurrentUser(int Id, string Username, string Role, long SessionId, string? IpAddress)
    {
        public bool IsSuperAdmin => Role == Roles.SuperAdmin;
    }

    public static class CurrentUserExtensions
    {
        public static CurrentUser GetCurrentUser(this HttpContext context)
        {
            var u = context.User;
            if (u.Identity?.IsAuthenticated != true) throw ApiException.Unauthorized();
            return new CurrentUser(
                int.Parse(u.FindFirstValue(TokenService.ClaimUserId)!),
                u.FindFirstValue(TokenService.ClaimUsername) ?? "",
                u.FindFirstValue(TokenService.ClaimRole) ?? "",
                long.Parse(u.FindFirstValue(TokenService.ClaimSessionId)!),
                context.Connection.RemoteIpAddress?.ToString());
        }
    }
}
