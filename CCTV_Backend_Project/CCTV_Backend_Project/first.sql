CREATE DATABASE IF NOT EXISTS cctv_ai_platform;
use cctv_ai_platform;

CREATE TABLE IF NOT EXISTS users (
  id            INT          NOT NULL AUTO_INCREMENT,
  employee_id   VARCHAR(64)  NOT NULL COMMENT 'Unique employee code entered at registration',
  name          VARCHAR(255) NOT NULL,
  created_at    DATETIME(6)  NOT NULL COMMENT 'UTC',
  PRIMARY KEY (id),
  UNIQUE KEY ix_users_employee_id (employee_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS face_embeddings (
  id              INT          NOT NULL AUTO_INCREMENT,
  user_id         INT          NOT NULL,
  embedding       BLOB         NOT NULL COMMENT '512 x float32, little-endian',
  thumbnail_path  VARCHAR(512) NOT NULL COMMENT 'File on disk under data/enrollment/',
  original_path   VARCHAR(512) NULL,
  face_width_px   INT          NOT NULL,
  det_score       DOUBLE       NOT NULL,
  created_at      DATETIME(6)  NOT NULL COMMENT 'UTC',
  PRIMARY KEY (id),
  KEY ix_face_embeddings_user_id (user_id),
  CONSTRAINT fk_face_embeddings_user
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS attendance_events (
  id              INT          NOT NULL AUTO_INCREMENT,
  user_id         INT          NOT NULL,
  camera_name     VARCHAR(255) NOT NULL,
  timestamp       DATETIME(6)  NOT NULL COMMENT 'UTC',
  confidence      DOUBLE       NOT NULL COMMENT 'Cosine similarity of the match',
  face_width_px   INT          NOT NULL,
  low_confidence  TINYINT(1)   NOT NULL DEFAULT 0,
  thumbnail_path  VARCHAR(512) NOT NULL,
  PRIMARY KEY (id),
  KEY ix_attendance_events_user_time (user_id, timestamp),
  KEY ix_attendance_events_time (timestamp),
  CONSTRAINT fk_attendance_events_user
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;



CREATE TABLE IF NOT EXISTS camera_restrictions (
  id          INT         NOT NULL AUTO_INCREMENT,
  user_id     INT         NOT NULL,
  camera_id   VARCHAR(64) NOT NULL COMMENT 'camera-01 | inside | corridor',
  created_at  DATETIME(6) NOT NULL COMMENT 'UTC',
  PRIMARY KEY (id),
  UNIQUE KEY ux_camera_restrictions_user_camera (user_id, camera_id),
  KEY ix_camera_restrictions_user_id (user_id),
  CONSTRAINT fk_camera_restrictions_user
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


CREATE TABLE IF NOT EXISTS activity_sessions (
  id             BIGINT       NOT NULL AUTO_INCREMENT,
  user_id        VARCHAR(64)  NULL     COMMENT 'Employee id (users.employee_id); NULL = unidentified',
  user_name      VARCHAR(255) NULL,
  camera_id      VARCHAR(64)  NOT NULL COMMENT 'camera-01 | inside | corridor',
  activity_type  VARCHAR(64)  NOT NULL COMMENT 'See activity_types',
  start_time     DATETIME(6)  NOT NULL COMMENT 'UTC',
  end_time       DATETIME(6)  NOT NULL COMMENT 'UTC',
  duration       DOUBLE       NOT NULL COMMENT 'Seconds',
  confidence     DOUBLE       NOT NULL DEFAULT 0 COMMENT 'Share of detection passes that agreed (0-1)',
  track_ids      VARCHAR(512) NULL     COMMENT 'Tracker ids that contributed (debug only)',
  created_at     DATETIME(6)  NOT NULL COMMENT 'UTC, when the row was written',
  PRIMARY KEY (id),
  KEY ix_activity_sessions_user_id (user_id),
  KEY ix_activity_sessions_camera_id (camera_id),
  KEY ix_activity_sessions_activity_type (activity_type),
  KEY ix_activity_sessions_start_time (start_time),
  KEY ix_activity_sessions_user_camera_start (user_id, camera_id, start_time),
  KEY ix_activity_sessions_type_start (activity_type, start_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


CREATE TABLE IF NOT EXISTS activity_events (
  id                BIGINT       NOT NULL AUTO_INCREMENT,
  camera_id         VARCHAR(64)  NOT NULL,
  track_id          INT          NOT NULL,
  person_id         VARCHAR(64)  NULL,
  person_name       VARCHAR(255) NULL,
  activity_type     VARCHAR(64)  NOT NULL,
  start_time        DATETIME(6)  NOT NULL,
  end_time          DATETIME(6)  NOT NULL,
  duration_seconds  DOUBLE       NOT NULL,
  confidence        DOUBLE       NOT NULL DEFAULT 0,
  created_at        DATETIME(6)  NOT NULL,
  PRIMARY KEY (id),
  KEY ix_activity_events_camera_id (camera_id),
  KEY ix_activity_events_track_id (track_id),
  KEY ix_activity_events_person_id (person_id),
  KEY ix_activity_events_activity_type (activity_type),
  KEY ix_activity_events_start_time (start_time),
  KEY ix_activity_person_camera_start (person_id, camera_id, start_time),
  KEY ix_activity_type_start (activity_type, start_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;



CREATE TABLE IF NOT EXISTS activity_types (
  code          VARCHAR(64)  NOT NULL,
  display_name  VARCHAR(64)  NOT NULL,
  category      VARCHAR(32)  NOT NULL COMMENT 'posture | movement | interaction | presence',
  sort_order    INT          NOT NULL,
  PRIMARY KEY (code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


INSERT IGNORE INTO activity_types (code, display_name, category, sort_order) VALUES
  ('SITTING',              'Sitting',      'posture',     1),
  ('STANDING',             'Standing',     'posture',     2),
  ('WALKING',              'Walking',      'movement',    3),
  ('STATIONARY',           'Low movement', 'movement',    4),
  ('PHONE_INTERACTION',    'Phone',        'interaction', 5),
  ('MONITOR_INTERACTION',  'Monitor',      'interaction', 6),
  ('COMPUTER_INTERACTION', 'Computer',     'interaction', 7),
  ('PRESENCE',             'Present',      'presence',    8);


CREATE TABLE IF NOT EXISTS cameras (
  camera_id     VARCHAR(64)  NOT NULL COMMENT 'Slug used everywhere else: camera-01 | inside | corridor',
  camera_no     INT          NOT NULL COMMENT 'Camera 1, 2, 3 as shown in the UI',
  name          VARCHAR(255) NOT NULL,
  ip_address    VARCHAR(64)  NULL,
  brand         VARCHAR(64)  NULL,
  is_active     TINYINT(1)   NOT NULL DEFAULT 1,
  PRIMARY KEY (camera_id),
  UNIQUE KEY ux_cameras_camera_no (camera_no)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


INSERT IGNORE INTO cameras (camera_id, camera_no, name, ip_address, brand) VALUES
  ('camera-01', 1, 'Camera 01', '172.29.7.7', 'Hikvision'),
  ('inside',    2, 'Inside',    '172.29.7.8', 'CP Plus'),
  ('corridor',  3, 'Corridor',  '172.29.7.6', 'Hikvision');


CREATE TABLE IF NOT EXISTS roles (
  code         VARCHAR(32)  NOT NULL,
  name         VARCHAR(64)  NOT NULL,
  rank_level   INT          NOT NULL COMMENT 'Higher = more senior role',
  description  VARCHAR(255) NOT NULL,
  PRIMARY KEY (code),
  UNIQUE KEY ux_roles_rank_level (rank_level)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


INSERT IGNORE INTO roles (code, name, rank_level, description) VALUES
  ('SUPER_ADMIN', 'Super Admin', 100, 'Highest role'),
  ('ADMIN',       'Admin',        50, 'Middle role'),
  ('VIEWER',      'Viewer',       10, 'Lowest role');


CREATE TABLE IF NOT EXISTS app_users (
  id                    INT          NOT NULL AUTO_INCREMENT,
  username              VARCHAR(64)  NOT NULL COMMENT 'Login name, case-insensitive (utf8mb4_unicode_ci)',
  full_name             VARCHAR(255) NOT NULL,
  email                 VARCHAR(255) NULL,
  password_hash         VARCHAR(255) NOT NULL COMMENT 'BCrypt/Argon2 hash. NEVER plain text',
  role_code             VARCHAR(32)  NOT NULL,
  all_cameras           TINYINT(1)   NOT NULL DEFAULT 0 COMMENT '1 = every camera, incl. ones added later. Ignored for SUPER_ADMIN (always all)',
  is_active             TINYINT(1)   NOT NULL DEFAULT 1 COMMENT '0 = blocked from logging in',
  must_change_password  TINYINT(1)   NOT NULL DEFAULT 1,
  failed_login_count    INT          NOT NULL DEFAULT 0,
  locked_until          DATETIME(6)  NULL     COMMENT 'UTC; set after too many failed logins',
  last_login_at         DATETIME(6)  NULL     COMMENT 'UTC',
  created_by            INT          NULL     COMMENT 'app_users.id of the creator; NULL = bootstrap',
  created_at            DATETIME(6)  NOT NULL COMMENT 'UTC',
  updated_at            DATETIME(6)  NOT NULL COMMENT 'UTC',
  PRIMARY KEY (id),
  UNIQUE KEY ux_app_users_username (username),
  UNIQUE KEY ux_app_users_email (email),
  KEY ix_app_users_role (role_code),
  CONSTRAINT fk_app_users_role
    FOREIGN KEY (role_code) REFERENCES roles (code),
  CONSTRAINT fk_app_users_created_by
    FOREIGN KEY (created_by) REFERENCES app_users (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


CREATE TABLE IF NOT EXISTS app_user_cameras (
  app_user_id  INT          NOT NULL,
  camera_id    VARCHAR(64)  NOT NULL,
  granted_by   INT          NULL COMMENT 'app_users.id',
  created_at   DATETIME(6)  NOT NULL COMMENT 'UTC',
  PRIMARY KEY (app_user_id, camera_id),
  KEY ix_app_user_cameras_camera (camera_id),
  CONSTRAINT fk_app_user_cameras_user
    FOREIGN KEY (app_user_id) REFERENCES app_users (id) ON DELETE CASCADE,
  CONSTRAINT fk_app_user_cameras_camera
    FOREIGN KEY (camera_id) REFERENCES cameras (camera_id) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT fk_app_user_cameras_granted_by
    FOREIGN KEY (granted_by) REFERENCES app_users (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS app_user_sessions (
  id           BIGINT       NOT NULL AUTO_INCREMENT,
  app_user_id  INT          NOT NULL,
  token_hash   CHAR(64)     NOT NULL COMMENT 'hex SHA-256 of the refresh token',
  created_at   DATETIME(6)  NOT NULL COMMENT 'UTC',
  expires_at   DATETIME(6)  NOT NULL COMMENT 'UTC',
  revoked_at   DATETIME(6)  NULL     COMMENT 'UTC; set on logout, password change, or disable',
  ip_address   VARCHAR(45)  NULL,
  user_agent   VARCHAR(255) NULL,
  PRIMARY KEY (id),
  UNIQUE KEY ux_app_user_sessions_token (token_hash),
  KEY ix_app_user_sessions_user (app_user_id, revoked_at),
  CONSTRAINT fk_app_user_sessions_user
    FOREIGN KEY (app_user_id) REFERENCES app_users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;



CREATE TABLE IF NOT EXISTS audit_log (
  id              BIGINT       NOT NULL AUTO_INCREMENT,
  at              DATETIME(6)  NOT NULL COMMENT 'UTC',
  app_user_id     INT          NULL     COMMENT 'Who did it; NULL = system',
  username        VARCHAR(64)  NULL     COMMENT 'Copied, so it survives the login being deleted',
  action          VARCHAR(64)  NOT NULL COMMENT 'e.g. login, login_failed, person.camera_access, login.create',
  target_type     VARCHAR(32)  NULL     COMMENT 'person | login | camera | settings',
  target_id       VARCHAR(64)  NULL,
  camera_id       VARCHAR(64)  NULL,
  details         JSON         NULL     COMMENT 'Before/after values. Never passwords or tokens',
  ip_address      VARCHAR(45)  NULL,
  PRIMARY KEY (id),
  KEY ix_audit_log_at (at),
  KEY ix_audit_log_user_at (app_user_id, at),
  KEY ix_audit_log_target (target_type, target_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =============================================================================
-- VIEWS -- per-person time for every activity, computed from activity_sessions
-- =============================================================================

-- Every session with the person's name and India-time columns.
CREATE OR REPLACE VIEW v_activity_sessions_named AS
SELECT
  s.id,
  s.user_id                                            AS employee_id,
  u.id                                                 AS user_db_id,
  COALESCE(u.name, s.user_name, 'Unknown')             AS person_name,
  s.camera_id,
  c.name                                               AS camera_name,
  s.activity_type,
  t.display_name                                       AS activity_name,
  s.start_time                                         AS start_time_utc,
  s.end_time                                           AS end_time_utc,
  CONVERT_TZ(s.start_time, '+00:00', '+05:30')         AS start_time_ist,
  CONVERT_TZ(s.end_time,   '+00:00', '+05:30')         AS end_time_ist,
  DATE(CONVERT_TZ(s.start_time, '+00:00', '+05:30'))   AS day_ist,
  s.duration                                           AS duration_seconds,
  s.confidence
FROM activity_sessions s
LEFT JOIN users          u ON u.employee_id = s.user_id
LEFT JOIN cameras        c ON c.camera_id   = s.camera_id
LEFT JOIN activity_types t ON t.code        = s.activity_type;

-- Per person, per day (India time), per camera: seconds of each activity.
CREATE OR REPLACE VIEW v_user_activity_daily AS
SELECT
  DATE(CONVERT_TZ(s.start_time, '+00:00', '+05:30'))   AS day_ist,
  s.user_id                                            AS employee_id,
  COALESCE(MAX(u.name), MAX(s.user_name), 'Unknown')   AS person_name,
  s.camera_id,
  ROUND(SUM(CASE WHEN s.activity_type = 'SITTING'              THEN s.duration ELSE 0 END), 1) AS sitting_seconds,
  ROUND(SUM(CASE WHEN s.activity_type = 'STANDING'             THEN s.duration ELSE 0 END), 1) AS standing_seconds,
  ROUND(SUM(CASE WHEN s.activity_type = 'WALKING'              THEN s.duration ELSE 0 END), 1) AS walking_seconds,
  ROUND(SUM(CASE WHEN s.activity_type = 'STATIONARY'           THEN s.duration ELSE 0 END), 1) AS low_movement_seconds,
  ROUND(SUM(CASE WHEN s.activity_type = 'PHONE_INTERACTION'    THEN s.duration ELSE 0 END), 1) AS phone_seconds,
  ROUND(SUM(CASE WHEN s.activity_type = 'MONITOR_INTERACTION'  THEN s.duration ELSE 0 END), 1) AS monitor_seconds,
  ROUND(SUM(CASE WHEN s.activity_type = 'COMPUTER_INTERACTION' THEN s.duration ELSE 0 END), 1) AS computer_seconds,
  ROUND(SUM(CASE WHEN s.activity_type = 'PRESENCE'             THEN s.duration ELSE 0 END), 1) AS present_seconds,
  SUM(CASE WHEN s.activity_type = 'PHONE_INTERACTION' THEN 1 ELSE 0 END)                       AS phone_sessions,
  MIN(CONVERT_TZ(s.start_time, '+00:00', '+05:30'))    AS first_seen_ist,
  MAX(CONVERT_TZ(s.end_time,   '+00:00', '+05:30'))    AS last_seen_ist
FROM activity_sessions s
LEFT JOIN users u ON u.employee_id = s.user_id
GROUP BY day_ist, s.user_id, s.camera_id;

-- Same, all cameras combined (one row per person per day).
CREATE OR REPLACE VIEW v_user_activity_daily_all AS
SELECT
  day_ist,
  employee_id,
  MAX(person_name)          AS person_name,
  SUM(sitting_seconds)      AS sitting_seconds,
  SUM(standing_seconds)     AS standing_seconds,
  SUM(walking_seconds)      AS walking_seconds,
  SUM(low_movement_seconds) AS low_movement_seconds,
  SUM(phone_seconds)        AS phone_seconds,
  SUM(monitor_seconds)      AS monitor_seconds,
  SUM(computer_seconds)     AS computer_seconds,
  SUM(present_seconds)      AS present_seconds,
  SUM(phone_sessions)       AS phone_sessions,
  MIN(first_seen_ist)       AS first_seen_ist,
  MAX(last_seen_ist)        AS last_seen_ist
FROM v_user_activity_daily
GROUP BY day_ist, employee_id;

-- Per person, all-time totals and session counts per activity.
CREATE OR REPLACE VIEW v_user_activity_totals AS
SELECT
  s.user_id                                            AS employee_id,
  COALESCE(MAX(u.name), MAX(s.user_name), 'Unknown')   AS person_name,
  s.activity_type,
  COUNT(*)                                             AS sessions,
  ROUND(SUM(s.duration), 1)                            AS total_seconds,
  ROUND(AVG(s.duration), 1)                            AS avg_session_seconds,
  ROUND(MAX(s.duration), 1)                            AS longest_session_seconds,
  MIN(s.start_time)                                    AS first_time_utc,
  MAX(s.end_time)                                      AS last_time_utc
FROM activity_sessions s
LEFT JOIN users u ON u.employee_id = s.user_id
GROUP BY s.user_id, s.activity_type;

-- First and last attendance punch per person per day (India time).
CREATE OR REPLACE VIEW v_attendance_daily AS
SELECT
  DATE(CONVERT_TZ(a.timestamp, '+00:00', '+05:30'))    AS day_ist,
  u.id                                                 AS user_db_id,
  u.employee_id,
  u.name                                               AS person_name,
  MIN(CONVERT_TZ(a.timestamp, '+00:00', '+05:30'))     AS first_punch_ist,
  MAX(CONVERT_TZ(a.timestamp, '+00:00', '+05:30'))     AS last_punch_ist,
  COUNT(*)                                             AS punches
FROM attendance_events a
JOIN users u ON u.id = a.user_id
GROUP BY day_ist, u.id, u.employee_id, u.name;

-- =============================================================================
-- ACCESS VIEWS -- what the backend checks on every request
-- =============================================================================

-- Effective cameras of each ACTIVE login: SUPER_ADMIN and all_cameras = 1 get
-- every active camera; everyone else gets only their app_user_cameras rows.
CREATE OR REPLACE VIEW v_app_user_cameras AS
SELECT a.id AS app_user_id, a.username, a.role_code, c.camera_id, c.name AS camera_name
FROM app_users a
JOIN cameras c ON c.is_active = 1
WHERE a.is_active = 1
  AND (a.role_code = 'SUPER_ADMIN' OR a.all_cameras = 1)
UNION ALL
SELECT a.id, a.username, a.role_code, c.camera_id, c.name
FROM app_users a
JOIN app_user_cameras ac ON ac.app_user_id = a.id
JOIN cameras c           ON c.camera_id    = ac.camera_id AND c.is_active = 1
WHERE a.is_active = 1
  AND a.role_code <> 'SUPER_ADMIN'
  AND a.all_cameras = 0;

-- Every tracked PERSON x every camera: may this person be seen there?
-- (Blocked = a camera_restrictions row exists -> "not welcome" alert.)
CREATE OR REPLACE VIEW v_person_camera_access AS
SELECT
  u.id                    AS user_db_id,
  u.employee_id,
  u.name                  AS person_name,
  c.camera_id,
  c.name                  AS camera_name,
  CASE WHEN r.id IS NULL THEN 'ALLOWED' ELSE 'BLOCKED' END AS access,
  r.created_at            AS blocked_since_utc
FROM users u
CROSS JOIN cameras c
LEFT JOIN camera_restrictions r
  ON r.user_id = u.id AND r.camera_id = c.camera_id;



