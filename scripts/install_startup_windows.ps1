# Database AI Assistant — Windows Auto-Start Setup
# Run: Right-click -> "Run with PowerShell", or:
#   powershell -ExecutionPolicy Bypass -File install_startup_windows.ps1
# Disable: Delete the shortcut from the Startup folder,
#   or run: Remove-Item "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\DatabaseAIAssistant.lnk"

$exePath = Join-Path $PSScriptRoot "..\DatabaseAIAssistant.exe" -Resolve
$startupDir = [Environment]::GetFolderPath('Startup')
$shortcutPath = Join-Path $startupDir "DatabaseAIAssistant.lnk"

$WshShell = New-Object -ComObject WScript.Shell
$shortcut = $WshShell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $exePath
$shortcut.WorkingDirectory = Split-Path $exePath -Parent
$shortcut.Description = "Database AI Assistant"
$shortcut.Save()

Write-Host "Auto-start enabled. Shortcut created at: $shortcutPath"
Write-Host "To disable: Delete the shortcut from the Startup folder."
Write-Host "  Remove-Item '$shortcutPath'"
