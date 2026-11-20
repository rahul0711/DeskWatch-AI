using CCTV_Backend_Project.Dtos;
using CCTV_Backend_Project.Security;
using CCTV_Backend_Project.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace CCTV_Backend_Project.Controllers
{
    /// <summary>Dashboard accounts (app_users). SUPER_ADMIN manages everyone; ADMIN
    /// manages VIEWERs only (rank rule, enforced in LoginService).</summary>
    [ApiController]
    [Route("api/db/logins")]
    public class LoginsController : ControllerBase
    {
        private readonly LoginService _logins;

        public LoginsController(LoginService logins) => _logins = logins;

        [HttpGet]
        [Authorize(Roles = Roles.AdminOrAbove)]
        public Task<List<LoginDto>> List([FromQuery] string? search, [FromQuery] string? role, [FromQuery] bool? active, CancellationToken ct)
            => _logins.ListAsync(HttpContext.GetCurrentUser(), search, role, active, ct);

        [HttpPost]
        [Authorize(Roles = Roles.AdminOrAbove)]
        public async Task<IActionResult> Create(CreateLoginRequest body, CancellationToken ct)
        {
            var created = await _logins.CreateAsync(HttpContext.GetCurrentUser(), body, ct);
            return StatusCode(StatusCodes.Status201Created, created);
        }

        [HttpPatch("{id:int}")]
        [Authorize(Roles = Roles.AdminOrAbove)]
        public Task<LoginDto> Update(int id, UpdateLoginRequest body, CancellationToken ct)
            => _logins.UpdateAsync(HttpContext.GetCurrentUser(), id, body, ct);

        [HttpPut("{id:int}/cameras")]
        [Authorize(Roles = Roles.AdminOrAbove)]
        public Task<LoginDto> SetCameras(int id, SetLoginCamerasRequest body, CancellationToken ct)
            => _logins.SetCamerasAsync(HttpContext.GetCurrentUser(), id, body, ct);

        [HttpPost("{id:int}/reset-password")]
        [Authorize(Roles = Roles.AdminOrAbove)]
        public async Task<IActionResult> ResetPassword(int id, ResetPasswordRequest body, CancellationToken ct)
        {
            await _logins.ResetPasswordAsync(HttpContext.GetCurrentUser(), id, body, ct);
            return Ok(new { ok = true });
        }

        [HttpDelete("{id:int}")]
        [Authorize(Roles = Roles.AdminOrAbove)]
        public async Task<IActionResult> Delete(int id, CancellationToken ct)
        {
            await _logins.DeleteAsync(HttpContext.GetCurrentUser(), id, ct);
            return Ok(new { ok = true });
        }
    }

    [ApiController]
    [Route("api/db/roles")]
    public class RolesController : ControllerBase
    {
        private readonly LoginService _logins;

        public RolesController(LoginService logins) => _logins = logins;

        /// <summary>Roles the caller may assign to a login.</summary>
        [HttpGet]
        [Authorize(Roles = Roles.AdminOrAbove)]
        public Task<List<RoleDto>> List(CancellationToken ct) => _logins.AssignableRolesAsync(HttpContext.GetCurrentUser(), ct);
    }
}
