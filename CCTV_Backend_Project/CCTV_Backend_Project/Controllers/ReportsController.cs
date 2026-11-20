using CCTV_Backend_Project.Dtos;
using CCTV_Backend_Project.Infrastructure;
using CCTV_Backend_Project.Security;
using CCTV_Backend_Project.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace CCTV_Backend_Project.Controllers
{
    [ApiController]
    [Route("api/db")]
    public class ReportsController : ControllerBase
    {
        private readonly ReportService _reports;

        public ReportsController(ReportService reports) => _reports = reports;

        /// <summary>Attendance punches; `date` = UTC day YYYY-MM-DD (as the Dashboard sends).</summary>
        [HttpGet("attendance")]
        [Authorize(Roles = Roles.Anyone)]
        public Task<List<AttendanceDto>> Attendance([FromQuery] string? date, CancellationToken ct)
            => _reports.AttendanceAsync(HttpContext.GetCurrentUser(), date, ct);

        /// <summary>Who came and who did not: every enrolled person for an India day
        /// (`date` YYYY-MM-DD, default today) with first/last punch.</summary>
        [HttpGet("attendance/daily")]
        [Authorize(Roles = Roles.AdminOrAbove)]
        public Task<List<AttendanceDailyDto>> AttendanceDaily([FromQuery] string? date, CancellationToken ct)
            => _reports.AttendanceDailyAsync(HttpContext.GetCurrentUser(), date, ct);

        /// <summary>Seconds of each activity per person; `date` = India day, default today.</summary>
        [HttpGet("activity/daily")]
        [Authorize(Roles = Roles.Anyone)]
        public Task<List<ActivityDailyDto>> ActivityDaily([FromQuery] string? date, [FromQuery] string? cameraId, CancellationToken ct)
            => _reports.ActivityDailyAsync(HttpContext.GetCurrentUser(), date, cameraId, ct);

        [HttpGet("activity/sessions")]
        [Authorize(Roles = Roles.Anyone)]
        public Task<Paged<ActivitySessionDto>> ActivitySessions(
            [FromQuery] string? date, [FromQuery] string? cameraId, [FromQuery] string? employeeId,
            [FromQuery] string? activityType, [FromQuery] int page = 1, [FromQuery] int pageSize = Paging.DefaultPageSize,
            CancellationToken ct = default)
            => _reports.ActivitySessionsAsync(HttpContext.GetCurrentUser(), date, cameraId, employeeId, activityType, page, pageSize, ct);

        [HttpGet("activity-types")]
        [Authorize(Roles = Roles.Anyone)]
        public Task<List<ActivityTypeDto>> ActivityTypes(CancellationToken ct) => _reports.ActivityTypesAsync(ct);
    }

    [ApiController]
    [Route("api/db/cameras")]
    public class CamerasController : ControllerBase
    {
        private readonly CameraService _cameras;

        public CamerasController(CameraService cameras) => _cameras = cameras;

        /// <summary>Cameras the caller may see (Super Admin: all, including inactive).</summary>
        [HttpGet]
        [Authorize(Roles = Roles.Anyone)]
        public Task<List<CameraDto>> List(CancellationToken ct) => _cameras.ListForAsync(HttpContext.GetCurrentUser(), ct);

        /// <summary>Rename / activate a camera in reports. The Python camera server
        /// still reads its cameras from .env.</summary>
        [HttpPatch("{cameraId}")]
        [Authorize(Roles = Roles.SuperAdmin)]
        public Task<CameraDto> Update(string cameraId, UpdateCameraRequest body, CancellationToken ct)
            => _cameras.UpdateAsync(HttpContext.GetCurrentUser(), cameraId, body, ct);

        /// <summary>Which analyses a camera runs: face, body movement, both or neither.</summary>
        [HttpPut("{cameraId}/features")]
        [Authorize(Roles = Roles.AdminOrAbove)]
        public Task<CameraDto> UpdateFeatures(string cameraId, UpdateCameraFeaturesRequest body, CancellationToken ct)
            => _cameras.UpdateFeaturesAsync(HttpContext.GetCurrentUser(), cameraId, body, ct);
    }

    [ApiController]
    [Route("api/db/audit")]
    public class AuditController : ControllerBase
    {
        private readonly AuditService _audit;

        public AuditController(AuditService audit) => _audit = audit;

        [HttpGet]
        [Authorize(Roles = Roles.SuperAdmin)]
        public Task<Paged<AuditDto>> List(
            [FromQuery] int? loginId, [FromQuery] string? action,
            [FromQuery] int page = 1, [FromQuery] int pageSize = Paging.DefaultPageSize, CancellationToken ct = default)
            => _audit.ListAsync(loginId, action, page, pageSize, ct);
    }

    [ApiController]
    [Route("api/db/health")]
    public class HealthController : ControllerBase
    {
        private readonly Db _db;

        public HealthController(Db db) => _db = db;

        [HttpGet]
        [AllowAnonymous]
        public async Task<IActionResult> Get(CancellationToken ct)
        {
            try
            {
                await using var conn = await _db.OpenAsync(ct);
                await using var cmd = conn.Command("SELECT 1");
                await cmd.ExecuteScalarAsync(ct);
                return Ok(new { status = "ok", database = "up" });
            }
            catch (MySql.Data.MySqlClient.MySqlException)
            {
                return StatusCode(503, new { status = "degraded", database = "down" });
            }
        }
    }
}
