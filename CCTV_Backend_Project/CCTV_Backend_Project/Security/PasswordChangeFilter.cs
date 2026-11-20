using CCTV_Backend_Project.Infrastructure;
using Microsoft.AspNetCore.Mvc.Filters;

namespace CCTV_Backend_Project.Security
{
    /// <summary>Marks the few actions a login may still call while it must change
    /// its password (me, change-password, logout).</summary>
    [AttributeUsage(AttributeTargets.Method)]
    public sealed class AllowWithPendingPasswordChangeAttribute : Attribute { }

    /// <summary>Global filter: a login with must_change_password = 1 can only call
    /// actions marked [AllowWithPendingPasswordChange].</summary>
    public class PasswordChangeFilter : IActionFilter
    {
        public void OnActionExecuting(ActionExecutingContext context)
        {
            var http = context.HttpContext;
            if (http.Items.TryGetValue(SessionValidator.MustChangePasswordItem, out var v) && v is true
                && !context.ActionDescriptor.EndpointMetadata.OfType<AllowWithPendingPasswordChangeAttribute>().Any())
            {
                throw new ApiException(403, "password_change_required", "Change your password first");
            }
        }

        public void OnActionExecuted(ActionExecutedContext context) { }
    }
}
