using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Security.Cryptography;
using System.Text;
using Microsoft.IdentityModel.Tokens;

namespace CCTV_Backend_Project.Security
{
    /// <summary>Issues HS256 JWTs from appsettings.json -> JwtSettings. The Python
    /// servers verify the same token with the same secret (.env JWT_SECRET).</summary>
    public class TokenService
    {
        public const string ClaimUserId = "sub";
        public const string ClaimUsername = "username";
        public const string ClaimRole = "role";
        public const string ClaimSessionId = "sid";

        private readonly JwtSettings _settings;

        public TokenService(JwtSettings settings) => _settings = settings;

        public TimeSpan Lifetime => TimeSpan.FromDays(_settings.ExpiryInDays);

        /// <summary>Returns the signed token and the jti whose SHA-256 is stored in
        /// app_user_sessions.token_hash.</summary>
        public string CreateToken(int userId, string username, string role, long sessionId, string jti, DateTime expiresUtc)
        {
            var claims = new[]
            {
                new Claim(ClaimUserId, userId.ToString()),
                new Claim(ClaimUsername, username),
                new Claim(ClaimRole, role),
                new Claim(ClaimSessionId, sessionId.ToString()),
                new Claim(JwtRegisteredClaimNames.Jti, jti),
            };
            var token = new JwtSecurityToken(
                issuer: _settings.Issuer,
                audience: _settings.Audience,
                claims: claims,
                notBefore: DateTime.UtcNow,
                expires: expiresUtc,
                signingCredentials: new SigningCredentials(_settings.SigningKey(), SecurityAlgorithms.HmacSha256));
            return new JwtSecurityTokenHandler().WriteToken(token);
        }

        public static string NewJti() => Convert.ToHexString(RandomNumberGenerator.GetBytes(32)).ToLowerInvariant();

        public static string Sha256Hex(string value)
            => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(value))).ToLowerInvariant();
    }

    public class JwtSettings
    {
        public string SecretKey { get; set; } = "";
        public string Issuer { get; set; } = "";
        public string Audience { get; set; } = "";
        public int ExpiryInDays { get; set; } = 1;

        public SymmetricSecurityKey SigningKey() => new(Encoding.UTF8.GetBytes(SecretKey));
    }
}
