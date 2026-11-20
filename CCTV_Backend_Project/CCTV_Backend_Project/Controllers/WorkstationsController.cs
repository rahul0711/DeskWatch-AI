using CCTV_Backend_Project.Dtos;
using CCTV_Backend_Project.Infrastructure;
using CCTV_Backend_Project.Security;
using CCTV_Backend_Project.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace CCTV_Backend_Project.Controllers
{
    /// <summary>Workstation zones per camera (Admin -> Workstations). Admin and
    /// Super Admin, for cameras in their scope; every change is audited.</summary>
    [ApiController]
    [Route("api/db/cameras/{cameraId}/workstations")]
    [Authorize(Roles = Roles.AdminOrAbove)]
    public class WorkstationsController : ControllerBase
    {
        private readonly WorkstationService _service;

        public WorkstationsController(WorkstationService service) => _service = service;

        [HttpGet]
        public Task<List<WorkstationDto>> List(string cameraId, CancellationToken ct)
            => _service.ListAsync(HttpContext.GetCurrentUser(), cameraId, ct);

        [HttpPost]
        public Task<WorkstationDto> Create(string cameraId, SaveWorkstationRequest body, CancellationToken ct)
            => _service.CreateAsync(HttpContext.GetCurrentUser(), cameraId, body, ct);

        [HttpPut("{id:int}")]
        public Task<WorkstationDto> Update(string cameraId, int id, SaveWorkstationRequest body, CancellationToken ct)
            => _service.UpdateAsync(HttpContext.GetCurrentUser(), cameraId, id, body, ct);

        [HttpDelete("{id:int}")]
        public async Task<IActionResult> Delete(string cameraId, int id, CancellationToken ct)
        {
            await _service.DeleteAsync(HttpContext.GetCurrentUser(), cameraId, id, ct);
            return NoContent();
        }
    }
}
