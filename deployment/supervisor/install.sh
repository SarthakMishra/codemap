#!/bin/bash

# Install CodeMap supervisor configuration

set -e

# Script must be run from the project root
if [[ ! -f "deployment/supervisor/codemap.conf" ]]; then
    echo "Error: This script must be run from the CodeMap project root directory."
    exit 1
fi

# Check if supervisor is installed
if ! command -v supervisorctl &> /dev/null; then
    echo "Error: Supervisor is not installed."
    echo "Please install supervisor first:"
    echo "  Ubuntu/Debian: sudo apt-get install supervisor"
    echo "  macOS: brew install supervisor"
    exit 1
fi

# Create log directory
LOGS_DIR="${HOME}/.codemap/logs"
mkdir -p "${LOGS_DIR}"

# Determine supervisor configuration location
if [[ -d "/etc/supervisor/conf.d" ]]; then
    # Debian/Ubuntu style
    SUPERVISOR_CONF_DIR="/etc/supervisor/conf.d"
    NEED_SUDO=true
elif [[ -d "/usr/local/etc/supervisor.d" ]]; then
    # Homebrew style (macOS)
    SUPERVISOR_CONF_DIR="/usr/local/etc/supervisor.d"
    NEED_SUDO=true
else
    # Fallback to user directory
    SUPERVISOR_CONF_DIR="${HOME}/.config/supervisor/conf.d"
    mkdir -p "${SUPERVISOR_CONF_DIR}"
    NEED_SUDO=false
fi

# Copy configuration file
echo "Installing CodeMap supervisor configuration..."
if [[ "${NEED_SUDO}" == "true" ]]; then
    sudo cp deployment/supervisor/codemap.conf "${SUPERVISOR_CONF_DIR}/codemap.conf"
    echo "Reloading supervisor configuration..."
    sudo supervisorctl reread
    sudo supervisorctl update
else
    cp deployment/supervisor/codemap.conf "${SUPERVISOR_CONF_DIR}/codemap.conf"
    echo "Reloading supervisor configuration..."
    supervisorctl reread
    supervisorctl update
fi

echo "CodeMap supervisor configuration installed successfully."
echo ""
echo "To start the service:"
if [[ "${NEED_SUDO}" == "true" ]]; then
    echo "  sudo supervisorctl start codemap"
    echo ""
    echo "To check status:"
    echo "  sudo supervisorctl status codemap"
    echo ""
    echo "To view logs:"
    echo "  sudo tail -f ${LOGS_DIR}/codemap_supervisor.out.log"
else
    echo "  supervisorctl start codemap"
    echo ""
    echo "To check status:"
    echo "  supervisorctl status codemap"
    echo ""
    echo "To view logs:"
    echo "  tail -f ${LOGS_DIR}/codemap_supervisor.out.log"
fi 