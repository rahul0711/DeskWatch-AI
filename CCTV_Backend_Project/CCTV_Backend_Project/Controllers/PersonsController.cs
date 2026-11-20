using CCTV_Backend_Project.Dtos;
using CCTV_Backend_Project.Security;
using CCTV_Backend_Project.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace CCTV_Backend_Project.Controllers
{
    /// <summary>Tracked people. Register / rename / delete / add faces stay on the
    /// Python attendance server (they need the face model).</summary>
    [ApiController]
    [Route("api/db/persons")]
    public class PersonsController : ControllerBase
    {
        private readonly PersonService _persons;

        public PersonsController(PersonService persons) => _persons = persons;

        [HttpGet]
        [Authorize(Roles = Roles.Anyone)]
        public Task<List<PersonDto>> List([FromQuery] string? search, CancellationToken ct)
            => _persons.ListAsync(HttpContext.GetCurrentUser(), search, ct);

        [HttpGet("{id:int}/camera-access")]
        [Authorize(Roles = Roles.Anyone)]
        public Task<List<PersonCameraAccessDto>> GetCameraAccess(int id, CancellationToken ct)
            => _persons.GetCameraAccessAsync(HttpContext.GetCurrentUser(), id, ct);

        /// <summary>Cameras this person must NOT be seen on (raises a "not welcome" alert).</summary>
        [HttpPut("{id:int}/camera-access")]
        [Authorize(Roles = Roles.AdminOrAbove)]
        public Task<CameraAccessResponse> SetCameraAccess(int id, CameraAccessRequest body, CancellationToken ct)
            => _persons.SetCameraAccessAsync(HttpContext.GetCurrentUser(), id, body, ct);
    }
}
