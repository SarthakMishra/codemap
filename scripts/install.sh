#!/bin/bash
# CodeMap Installation Script
# WARNING: This tool is under active development and testing.
# Use with caution.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Ensure script can read user input
exec < /dev/tty || true

# Print warning
echo -e "${YELLOW}WARNING: CodeMap is currently in active development and testing phase.${NC}"
echo -e "${YELLOW}Use with caution in production environments.${NC}\n"

# Check for root/sudo privileges
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Error: This script requires root privileges to install globally.${NC}"
    echo -e "Please run with sudo: ${GREEN}sudo curl -LsSf https://raw.githubusercontent.com/SarthakMishra/codemap/main/install.sh | sudo bash${NC}"
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

# Create temporary directory for cloning
TEMP_DIR=$(mktemp -d)
echo -e "${GREEN}Creating temporary directory at $TEMP_DIR...${NC}"

# Cleanup function
cleanup() {
    echo -e "${YELLOW}Cleaning up temporary files...${NC}"
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

echo -e "${GREEN}Cloning CodeMap repository...${NC}"
git clone https://github.com/SarthakMishra/codemap.git "$TEMP_DIR"
cd "$TEMP_DIR"

echo -e "${GREEN}Installing CodeMap globally...${NC}"
if command -v pip3 &> /dev/null; then
    pip3 install --upgrade pip
    pip3 install .
else
    python3 -m pip install --upgrade pip
    python3 -m pip install .
fi

echo -e "\n${GREEN}Installation complete!${NC}"
echo -e "${YELLOW}Note: CodeMap is in development. Please use with caution.${NC}"
echo -e "\nYou can now use CodeMap from anywhere:"
echo -e "   ${GREEN}codemap generate /path/to/your/project${NC}"

# Verify installation
if command -v codemap &> /dev/null; then
    echo -e "\n${GREEN}CodeMap was successfully installed globally!${NC}"
else
    echo -e "\n${RED}Warning: Installation completed but 'codemap' command not found in PATH.${NC}"
    echo -e "You might need to restart your terminal or add Python's bin directory to your PATH."
fi 