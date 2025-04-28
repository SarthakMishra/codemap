#!/bin/bash

# Install CodeMap systemd service for the current user

set -e

# Script must be run from the project root
if [[ ! -f "deployment/systemd/codemap.service" ]]; then
    echo "Error: This script must be run from the CodeMap project root directory."
    exit 1
fi

# Create user systemd directory if it doesn't exist
SYSTEMD_DIR="${HOME}/.config/systemd/user"
mkdir -p "${SYSTEMD_DIR}"

# Copy service file
echo "Installing CodeMap systemd service..."
cp deployment/systemd/codemap.service "${SYSTEMD_DIR}/"

# Reload systemd user configuration
systemctl --user daemon-reload

echo "CodeMap service installed successfully."
echo ""
echo "To enable the service to start automatically (at login):"
echo "  systemctl --user enable codemap.service"
echo ""
echo "To start the service now:"
echo "  systemctl --user start codemap.service"
echo ""
echo "To check status:"
echo "  systemctl --user status codemap.service"
echo ""
echo "To view logs:"
echo "  journalctl --user -u codemap.service" 