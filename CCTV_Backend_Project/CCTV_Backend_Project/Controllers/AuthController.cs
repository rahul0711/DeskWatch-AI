using CCTV_Backend_Project.Dtos;
using CCTV_Backend_Project.Security;
using CCTV_Backend_Project.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.RateLimiting;

namespace CCTV_Backend_Project.Controllers
{
    [ApiController]
    [Route("api/db/auth")]
    public class AuthController : ControllerBase
    {
        private readonly AuthService _auth;

        public AuthController(AuthService auth) => _auth = auth;

        /// <summary>Username or email + password. Returns the bearer token for every
        /// other call (this backend and the Python servers).</summary>
        [HttpPost("login")]
        [AllowAnonymous]
        [EnableRateLimiting("login")]
        public Task<LoginResponse> Login(LoginRequest body, CancellationToken ct)
            => _auth.LoginAsync(body, HttpContext.Connection.RemoteIpAddress?.ToString(), Request.Headers.UserAgent.ToString(), ct);

        [HttpGet("me")]
        [Authorize(Roles = Roles.Anyone)]
        [AllowWithPendingPasswordChange]
        public Task<MeResponse> Me(CancellationToken ct) => _auth.MeAsync(HttpContext.GetCurrentUser(), ct);

        [HttpPost("change-password")]
        [Authorize(Roles = Roles.Anyone)]
        [AllowWithPendingPasswordChange]
        public async Task<MeResponse> ChangePassword(ChangePasswordRequest body, CancellationToken ct)
        {
            var user = HttpContext.GetCurrentUser();
            await _auth.ChangePasswordAsync(user, body, ct);
            return await _auth.MeAsync(user, ct);
        }

        [HttpPost("logout")]
        [Authorize(Roles = Roles.Anyone)]
        [AllowWithPendingPasswordChange]
        public async Task<IActionResult> Logout(CancellationToken ct)
        {
            await _auth.LogoutAsync(HttpContext.GetCurrentUser(), ct);
            return Ok(new { ok = true });
        }
    }
}
