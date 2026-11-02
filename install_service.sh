#!/usr/bin/env bash
# =====================================================================
# Auto-configure and install systemd background service for Robot Assistant
# Run on Raspberry Pi: bash install_service.sh
# =====================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_USER="${USER:-$(whoami)}"

echo "⚙️ Configuring Robot Assistant systemd service..."
echo "  • Project Directory: $PROJECT_DIR"
echo "  • Run As User      : $CURRENT_USER"

SERVICE_FILE="/etc/systemd/system/robot-assistant.service"

sudo bash -c "cat > '$SERVICE_FILE'" <<EOF
[Unit]
Description=Raspberry Pi Robot Assistant Headless Service
Documentation=https://github.com/vinaykumarsrinivas1980-blip/up-robo
Wants=network-online.target sound.target
After=network-online.target sound.target multi-user.target

[Service]
Type=simple
User=${CURRENT_USER}
WorkingDirectory=${PROJECT_DIR}
ExecStart=${PROJECT_DIR}/venv/bin/python main.py --wake-word

# Environment variables setup
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=-${PROJECT_DIR}/.env

# Restart policies (restarts on crash after 10s)
Restart=on-failure
RestartSec=10s

# Logging configuration
StandardOutput=journal
StandardError=journal
SyslogIdentifier=robot-assistant

# Signal & graceful stop configuration
KillSignal=SIGTERM
TimeoutStopSec=15s

[Install]
WantedBy=multi-user.target
EOF

echo "🔄 Reloading systemd daemon..."
sudo systemctl daemon-reload

echo "🔌 Enabling robot-assistant to start automatically on Raspberry Pi boot..."
sudo systemctl enable robot-assistant.service

echo "🚀 Starting robot-assistant service now..."
sudo systemctl restart robot-assistant.service

echo ""
echo "====================================================================="
echo "✅ Robot Assistant background service installed and started!"
echo "Useful commands on your Raspberry Pi:"
echo "  • Check service status : sudo systemctl status robot-assistant"
echo "  • View live logs       : journalctl -u robot-assistant -f"
echo "  • Stop the assistant   : sudo systemctl stop robot-assistant"
echo "  • Restart assistant    : sudo systemctl restart robot-assistant"
echo "====================================================================="
