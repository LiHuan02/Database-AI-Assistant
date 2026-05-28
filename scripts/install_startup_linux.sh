#!/bin/bash
# Database AI Assistant — Linux Auto-Start Setup
# Run: bash install_startup_linux.sh
# Disable: rm ~/.config/autostart/database-ai-assistant.desktop

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
EXEC_PATH="$APP_DIR/DatabaseAIAssistant"
DESKTOP_FILE="$HOME/.config/autostart/database-ai-assistant.desktop"

mkdir -p "$HOME/.config/autostart"

cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Type=Application
Name=Database AI Assistant
Comment=智能知识库问答助手
Exec=$EXEC_PATH
Path=$APP_DIR
Terminal=false
X-GNOME-Autostart-enabled=true
EOF

echo "Auto-start enabled: $DESKTOP_FILE"
echo "To disable: rm $DESKTOP_FILE"
