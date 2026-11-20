using System.Text.Json;
using System.Threading.RateLimiting;
using CCTV_Backend_Project.Infrastructure;
using CCTV_Backend_Project.Security;
using CCTV_Backend_Project.Services;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.IdentityModel.Tokens;

var builder = WebApplication.CreateBuilder(args);

var jwt = builder.Configuration.GetSection("JwtSettings").Get<JwtSettings>() ?? new JwtSettings();
if (jwt.SecretKey.Length < 32)
    throw new InvalidOperationException("JwtSettings:SecretKey in appsettings.json must be at least 32 characters");
builder.Services.AddSingleton(jwt);

// API only: controllers, no views. JSON is snake_case to match the Python servers.
builder.Services.AddControllers(o => o.Filters.Add<PasswordChangeFilter>())
    .AddJsonOptions(o => o.JsonSerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower)
    .ConfigureApiBehaviorOptions(o =>
    {
        // Model-binding errors in the same { detail } shape as every other error.
        o.InvalidModelStateResponseFactory = ctx =>
        {
            var first = ctx.ModelState.Where(e => e.Value?.Errors.Count > 0)
                .Select(e => $"{e.Key}: {e.Value!.Errors[0].ErrorMessage}").FirstOrDefault() ?? "Invalid request";
            var problem = new Microsoft.AspNetCore.Mvc.ProblemDetails { Status = 400, Title = "bad_request", Detail = first };
            problem.Extensions["code"] = "bad_request";
            return new Microsoft.AspNetCore.Mvc.BadRequestObjectResult(problem);
        };
    });

builder.Services.AddMemoryCache();
builder.Services.AddSingleton<Db>();
builder.Services.AddSingleton<TokenService>();
builder.Services.AddSingleton<SessionValidator>();
builder.Services.AddScoped<AuditService>();
builder.Services.AddScoped<CameraService>();
builder.Services.AddScoped<AuthService>();
builder.Services.AddScoped<LoginService>();
builder.Services.AddScoped<PersonService>();
builder.Services.AddScoped<ReportService>();
builder.Services.AddScoped<WorkstationService>();

builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(o =>
    {
        o.MapInboundClaims = false; // keep "sub", "role", "sid" as written
        o.TokenValidationParameters = new TokenValidationParameters
        {
            ValidateIssuer = true,
            ValidIssuer = jwt.Issuer,
            ValidateAudience = true,
            ValidAudience = jwt.Audience,
            ValidateIssuerSigningKey = true,
            IssuerSigningKey = jwt.SigningKey(),
            ValidateLifetime = true,
            ClockSkew = TimeSpan.FromMinutes(1),
            NameClaimType = TokenService.ClaimUsername,
            RoleClaimType = TokenService.ClaimRole,
        };
        o.Events = new JwtBearerEvents
        {
            // A valid signature is not enough: the session must still be open and
            // the login active (logout / disable take effect immediately).
            OnTokenValidated = async ctx =>
            {
                var validator = ctx.HttpContext.RequestServices.GetRequiredService<SessionValidator>();
                var error = await validator.ValidateAsync(ctx.Principal!, ctx.HttpContext, ctx.HttpContext.RequestAborted);
                if (error is not null) ctx.Fail(error);
            },
            OnChallenge = async ctx =>
            {
                ctx.HandleResponse();
                await ApiExceptionMiddleware.Write(ctx.HttpContext, 401, "unauthorized",
                    ctx.AuthenticateFailure?.Message is { Length: > 0 } m && !m.StartsWith("IDX") ? m : "Not authenticated");
            },
            OnForbidden = ctx => ApiExceptionMiddleware.Write(ctx.HttpContext, 403, "forbidden", "Your role is not allowed to do this"),
        };
    });
builder.Services.AddAuthorization();

// Brute-force guard on the login endpoint: 10 attempts per minute per IP.
builder.Services.AddRateLimiter(o =>
{
    o.RejectionStatusCode = StatusCodes.Status429TooManyRequests;
    o.OnRejected = (ctx, _) => new ValueTask(ApiExceptionMiddleware.Write(
        ctx.HttpContext, 429, "too_many_requests", "Too many login attempts, wait a minute and try again"));
    o.AddPolicy("login", http => RateLimitPartition.GetFixedWindowLimiter(
        http.Connection.RemoteIpAddress?.ToString() ?? "unknown",
        _ => new FixedWindowRateLimiterOptions { PermitLimit = 10, Window = TimeSpan.FromMinutes(1) }));
});

// The React dev server normally reaches this API through its /api/db proxy
// (same origin); CORS covers calling it directly from the frontend's host.
builder.Services.AddCors(o => o.AddDefaultPolicy(p => p
    .SetIsOriginAllowed(origin => Uri.TryCreate(origin, UriKind.Absolute, out var uri) && uri.Port is 5173 or 5174)
    .AllowAnyHeader()
    .AllowAnyMethod()));

var app = builder.Build();

app.UseMiddleware<ApiExceptionMiddleware>();
app.UseCors();
app.UseRateLimiter();
app.UseAuthentication();
app.UseAuthorization();
app.MapControllers();

app.Run();
