using Microsoft.AspNetCore.Mvc;
using MySql.Data.MySqlClient;

namespace CCTV_Backend_Project.Infrastructure
{
    /// <summary>Thrown by services; turned into an RFC 7807 response by ApiExceptionMiddleware.
    /// `detail` is what the frontend shows (same field FastAPI uses).</summary>
    public class ApiException : Exception
    {
        public int Status { get; }
        public string Code { get; }

        public ApiException(int status, string code, string detail) : base(detail)
        {
            Status = status;
            Code = code;
        }

        public static ApiException BadRequest(string detail) => new(400, "bad_request", detail);
        public static ApiException Unauthorized(string detail = "Not authenticated") => new(401, "unauthorized", detail);
        public static ApiException Forbidden(string detail = "You are not allowed to do this") => new(403, "forbidden", detail);
        public static ApiException OutOfScope(string cameraId) => new(403, "out_of_scope", $"You do not have access to camera '{cameraId}'");
        public static ApiException NotFound(string detail) => new(404, "not_found", detail);
        public static ApiException Conflict(string detail) => new(409, "conflict", detail);
    }

    public class ApiExceptionMiddleware
    {
        private readonly RequestDelegate _next;
        private readonly ILogger<ApiExceptionMiddleware> _logger;

        public ApiExceptionMiddleware(RequestDelegate next, ILogger<ApiExceptionMiddleware> logger)
        {
            _next = next;
            _logger = logger;
        }

        public async Task InvokeAsync(HttpContext context)
        {
            try
            {
                await _next(context);
            }
            catch (ApiException ex)
            {
                await Write(context, ex.Status, ex.Code, ex.Message);
            }
            catch (MySqlException ex) when (ex.Number == 1062)
            {
                // Unique key: username / email / employee id already used.
                await Write(context, 409, "duplicate", "That value is already in use");
            }
            catch (MySqlException ex)
            {
                // Never echo SQL or the connection string back.
                _logger.LogError("MySQL error {Number}: {Message}", ex.Number, ex.Message);
                await Write(context, 503, "database_unavailable", "Database error, try again");
            }
            catch (OperationCanceledException) when (context.RequestAborted.IsCancellationRequested)
            {
                // Client went away; nothing to send.
            }
        }

        public static Task Write(HttpContext context, int status, string code, string detail)
        {
            if (context.Response.HasStarted) return Task.CompletedTask;
            context.Response.Clear();
            context.Response.StatusCode = status;
            var problem = new ProblemDetails { Status = status, Title = code, Detail = detail };
            problem.Extensions["code"] = code;
            return context.Response.WriteAsJsonAsync(problem, (System.Text.Json.JsonSerializerOptions?)null, "application/problem+json");
        }
    }
}
