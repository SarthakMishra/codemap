#!/bin/bash
# CodeMap Upgrade Script
# Updates an existing CodeMap installation to the latest version.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if script is being run through pipe
if [ ! -t 0 ] && [ -z "$BASH_EXECUTION_STRING" ]; then
    echo -e "${RED}Error: This script needs to be run with bash -s.${NC}"
    echo -e "Please run: ${GREEN}curl -LsSf https://raw.githubusercontent.com/SarthakMishra/codemap/main/scripts/upgrade.sh | sudo bash -s${NC}"
    exit 1
fi

# Print warning
echo -e "${YELLOW}WARNING: CodeMap is currently in active development and testing phase.${NC}"
echo -e "${YELLOW}Upgrading might introduce breaking changes. Please backup your configuration.${NC}\n"

# Check for root/sudo privileges
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Error: This script requires root privileges to upgrade globally.${NC}"
    echo -e "Please run with sudo: ${GREEN}sudo curl -LsSf https://raw.githubusercontent.com/SarthakMishra/codemap/main/upgrade.sh | sudo bash${NC}"
    exit 1
fi

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required but not installed.${NC}"
    echo -e "Would you like to continue after installing Python 3? (y/N) "
    read -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    exit 1
fi

# Check for pip
if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
    echo -e "${RED}Error: pip3 is not installed.${NC}"
    echo -e "Would you like to continue after installing pip? (y/N) "
    read -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    exit 1
fi

# Check if CodeMap is installed
if ! command -v codemap &> /dev/null; then
    echo -e "${RED}Error: CodeMap is not installed.${NC}"
    echo -e "Please install CodeMap first using the installation script:"
    echo -e "${GREEN}sudo curl -LsSf https://raw.githubusercontent.com/SarthakMishra/codemap/main/install.sh | sudo bash${NC}"
    exit 1
fi

# Create temporary directory for cloning
TEMP_DIR=$(mktemp -d)
echo -e "${GREEN}Creating temporary directory at $TEMP_DIR...${NC}"

# Cleanup function
cleanup() {
    echo -e "${YELLOW}Cleaning up temporary files...${NC}"
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Backup existing configuration
CONFIG_DIR="$HOME/.codemap"
if [ -d "$CONFIG_DIR" ]; then
    BACKUP_DIR="${CONFIG_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    echo -e "${GREEN}Backing up existing configuration to $BACKUP_DIR...${NC}"
    cp -r "$CONFIG_DIR" "$BACKUP_DIR"
fi

echo -e "${GREEN}Cloning latest CodeMap version...${NC}"
git clone https://github.com/SarthakMishra/codemap.git "$TEMP_DIR"
cd "$TEMP_DIR"

echo -e "${GREEN}Upgrading CodeMap...${NC}"
if command -v pip3 &> /dev/null; then
    pip3 install --upgrade pip
    pip3 install --upgrade .
else
    python3 -m pip install --upgrade pip
    python3 -m pip install --upgrade .
fi

# Verify upgrade
CURRENT_VERSION=$(codemap --version 2>/dev/null || echo "unknown")
echo -e "\n${GREEN}CodeMap has been upgraded!${NC}"
echo -e "Current version: ${YELLOW}$CURRENT_VERSION${NC}"

# Restore configuration if backup exists
if [ -d "$BACKUP_DIR" ]; then
    echo -e "\n${YELLOW}Note: Your previous configuration was backed up to:${NC}"
    echo -e "${GREEN}$BACKUP_DIR${NC}"
    echo -e "If the upgrade causes any issues, you can restore it manually."
fi

echo -e "\n${GREEN}Upgrade complete!${NC}"
echo -e "${YELLOW}Note: If you experience any issues after upgrading, please:${NC}"
echo -e "1. Check the changelog at: ${GREEN}https://github.com/SarthakMishra/codemap/blob/main/CHANGELOG.md${NC}"
echo -e "2. Report issues at: ${GREEN}https://github.com/SarthakMishra/codemap/issues${NC}"
echo -e "3. Restore your backup configuration if needed" 