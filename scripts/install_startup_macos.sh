#!/bin/bash
# Database AI Assistant — macOS Auto-Start Setup
# Run: bash install_startup_macos.sh
# Disable: launchctl unload ~/Library/LaunchAgents/com.database-ai-assistant.plist
#          rm ~/Library/LaunchAgents/com.database-ai-assistant.plist

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
EXEC_PATH="$APP_DIR/DatabaseAIAssistant"
PLIST_FILE="$HOME/Library/LaunchAgents/com.database-ai-assistant.plist"

mkdir -p "$HOME/Library/LaunchAgents"

cat > "$PLIST_FILE" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.database-ai-assistant</string>
    <key>ProgramArguments</key>
    <array>
        <string>$EXEC_PATH</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$APP_DIR</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
</dict>
</plist>
EOF

launchctl load "$PLIST_FILE"

echo "Auto-start enabled: $PLIST_FILE"
echo "To disable: launchctl unload $PLIST_FILE && rm $PLIST_FILE"
