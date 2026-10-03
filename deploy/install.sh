#!/usr/bin/env bash
# One-command deploy on any Linux PC: builds the project (deploy/setup.sh),
# installs the backend as a systemd service (auto-starts on boot, restarts on
# crash) and puts nginx on NGINX_PORT in front of it. Works from wherever the
# project folder lives -- no paths are hardcoded.
#
#   sudo bash deploy/install.sh
#   sudo NGINX_PORT=9005 BACKEND_PORT=8001 bash deploy/install.sh   # defaults
set -euo pipefail

NGINX_PORT="${NGINX_PORT:-9005}"
BACKEND_PORT="${BACKEND_PORT:-8001}"
SERVICE=attendance-backend
SITE="attendance-$NGINX_PORT"

fail() { echo "ERROR: $*" >&2; exit 1; }

[ "$EUID" -eq 0 ] || fail "Run with sudo: sudo bash deploy/install.sh"
command -v nginx >/dev/null || fail "nginx not installed. Run: sudo apt install -y nginx"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(stat -c %U "$PROJECT_DIR")}"
[ "$RUN_USER" != "root" ] || fail "Could not determine a non-root user to run the service as."
RUN_GROUP="$(id -gn "$RUN_USER")"

# Refuse to steal a port another nginx site already listens on.
if grep -rlE "listen[^;]*\b$NGINX_PORT\b" /etc/nginx/sites-enabled/ /etc/nginx/conf.d/ 2>/dev/null \
    | grep -vE "/(attendance-[0-9]+)$" | grep -q .; then
  fail "Port $NGINX_PORT is already used by another nginx site. Pick another: sudo NGINX_PORT=90xx bash deploy/install.sh"
fi

echo "==> Building project as $RUN_USER in $PROJECT_DIR"
sudo -u "$RUN_USER" -H bash "$PROJECT_DIR/deploy/setup.sh"

echo "==> Installing systemd service ($SERVICE, 127.0.0.1:$BACKEND_PORT)"
CUDA_LIBS="$(sudo -u "$RUN_USER" bash "$PROJECT_DIR/deploy/cuda_lib_path.sh")"
cat > "/etc/systemd/system/$SERVICE.service" <<EOF
[Unit]
Description=CCTV Attendance backend (FastAPI/uvicorn, SCRFD+AdaFace+FAISS)
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
User=$RUN_USER
Group=$RUN_GROUP
WorkingDirectory=$PROJECT_DIR
Environment="LD_LIBRARY_PATH=$CUDA_LIBS"
ExecStart="$PROJECT_DIR/.venv/bin/python" -m uvicorn app.web.attendance_server:app --host 127.0.0.1 --port $BACKEND_PORT
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable "$SERVICE"
systemctl restart "$SERVICE"

echo "==> Installing nginx site $SITE (port $NGINX_PORT)"
# Drop any earlier attendance-<port> site so the app is served on one port only.
for old in /etc/nginx/sites-enabled/attendance-* /etc/nginx/sites-available/attendance-*; do
  [ -e "$old" ] || [ -L "$old" ] || continue
  [ "$(basename "$old")" = "$SITE" ] && continue
  echo "    removing old site $old"
  rm -f "$old"
done
sed -e "s/__NGINX_PORT__/$NGINX_PORT/g" -e "s/__BACKEND_PORT__/$BACKEND_PORT/g" \
  "$PROJECT_DIR/deploy/nginx-attendance.conf.template" > "/etc/nginx/sites-available/$SITE"
ln -sf "/etc/nginx/sites-available/$SITE" "/etc/nginx/sites-enabled/$SITE"
nginx -t
systemctl enable nginx >/dev/null 2>&1 || true
systemctl reload nginx || systemctl restart nginx

if command -v ufw >/dev/null && ufw status 2>/dev/null | grep -q "Status: active"; then
  echo "==> Opening port $NGINX_PORT in ufw"
  ufw allow "$NGINX_PORT/tcp"
fi

echo "==> Waiting for backend to come up (model loading)..."
for _ in $(seq 1 60); do
  curl -fs -o /dev/null "http://127.0.0.1:$NGINX_PORT/" && break
  sleep 1
done

echo
systemctl --no-pager status "$SERVICE" | head -6
echo
for ip in $(hostname -I); do
  case "$ip" in *:*) ;; *) echo "Dashboard: http://$ip:$NGINX_PORT/";; esac
done
echo "Logs:      sudo journalctl -u $SERVICE -f"
