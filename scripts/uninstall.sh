#!/bin/bash
# CodeMap Uninstall Script
# Removes CodeMap and its dependencies from your system.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print warning
echo -e "${YELLOW}WARNING: This script will remove CodeMap from your system.${NC}"
echo -e "${YELLOW}This includes:${NC}"
echo -e "  - The CodeMap package"
echo -e "  - Generated documentation files"
echo -e "  - Cache directories"
echo -e "  - Configuration files"
echo -e "\n${YELLOW}Your source code and git repository will not be affected.${NC}\n"

# Check for root/sudo privileges
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Error: This script requires root privileges to uninstall globally.${NC}"
    echo -e "Please run with sudo: ${GREEN}sudo ./uninstall.sh${NC}"
    exit 1
fi

# Prompt for confirmation
read -p "Are you sure you want to proceed? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Uninstall cancelled.${NC}"
    exit 0
fi

echo -e "\n${GREEN}Starting uninstall...${NC}"

# Remove the package
echo -e "${GREEN}Removing CodeMap package...${NC}"
pip3 uninstall -y codemap || true

# Remove configuration and cache directories
echo -e "${GREEN}Removing configuration and cache files...${NC}"
rm -rf ~/.codemap ~/.codemap_cache 2>/dev/null || true

# Remove documentation files (only in current directory)
echo -e "${GREEN}Cleaning up documentation files...${NC}"
find . -maxdepth 2 -type f -name "*.code-map.*.md" -delete 2>/dev/null || true
rm -rf ./documentation 2>/dev/null || true

# Remove build artifacts
echo -e "${GREEN}Removing build artifacts...${NC}"
rm -rf build/ dist/ *.egg-info/ .eggs/ 2>/dev/null || true
rm -f build/languages.so 2>/dev/null || true

# Remove cache directories
echo -e "${GREEN}Removing cache directories...${NC}"
rm -rf .pytest_cache/ .ruff_cache/ __pycache__/ 2>/dev/null || true

# Remove virtual environment if it exists
if [ -d ".venv" ]; then
    echo -e "${GREEN}Removing virtual environment...${NC}"
    rm -rf .venv/
fi

echo -e "\n${GREEN}CodeMap has been successfully uninstalled!${NC}"
echo -e "${YELLOW}Note: If you installed CodeMap in other locations, you may need to remove those installations manually.${NC}" 