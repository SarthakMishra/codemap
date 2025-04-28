#!/usr/bin/env bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'
NORMAL='\033[0m'

# Repository URL
REPO_URL="https://github.com/SarthakMishra/codemap.git"

print_header() {
    echo -e "${BOLD}${GREEN}"
    echo "  ██████╗ ██████╗ ██████╗ ███████╗███╗   ███╗ █████╗ ██████╗ "
    echo " ██╔════╝██╔═══██╗██╔══██╗██╔════╝████╗ ████║██╔══██╗██╔══██╗"
    echo " ██║     ██║   ██║██║  ██║█████╗  ██╔████╔██║███████║██████╔╝"
    echo " ██║     ██║   ██║██║  ██║██╔══╝  ██║╚██╔╝██║██╔══██║██╔═══╝ "
    echo " ╚██████╗╚██████╔╝██████╔╝███████╗██║ ╚═╝ ██║██║  ██║██║     "
    echo "  ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝     "
    echo -e "${NC}${NORMAL}"
    echo -e "${BOLD}CodeMap Installer${NORMAL}"
    echo "------------------------------------------------"
}

check_python_version() {
    echo -e "Checking Python version..."
    if command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}Error: Python is not installed. Please install Python 3.12 or higher.${NC}"
        exit 1
    fi

    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    echo -e "Found Python $PYTHON_VERSION"

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]); then
        echo -e "${YELLOW}Warning: CodeMap recommends Python 3.12 or higher.${NC}"
        echo -e "Current version: $PYTHON_VERSION"
        
        read -p "Continue with Python $PYTHON_VERSION? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${RED}Installation aborted. Please install Python 3.12 or higher.${NC}"
            exit 1
        fi
    fi
}

check_pip() {
    echo -e "Checking pip installation..."
    if ! $PYTHON_CMD -m pip --version &>/dev/null; then
        echo -e "${YELLOW}pip not found. Installing pip...${NC}"
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        $PYTHON_CMD get-pip.py --user
        rm get-pip.py
    fi
    echo -e "${GREEN}pip is installed.${NC}"
}

install_codemap() {
    echo -e "Installing CodeMap..."
    
    # Install using pip with --user flag for user-specific installation
    $PYTHON_CMD -m pip install --user "git+$REPO_URL"
    
    # Check if the installation was successful
    if $PYTHON_CMD -m pip show codemap &>/dev/null; then
        echo -e "${GREEN}CodeMap installed successfully!${NC}"
        CODEMAP_VERSION=$($PYTHON_CMD -m pip show codemap | grep -i version | cut -d ' ' -f 2)
        echo -e "Version: $CODEMAP_VERSION"
        
        # Ensure PATH includes user's bin directory
        USER_BIN_DIR=$($PYTHON_CMD -m site --user-base)/bin
        
        if [[ ":$PATH:" != *":$USER_BIN_DIR:"* ]]; then
            echo -e "${YELLOW}Note: You may need to add $USER_BIN_DIR to your PATH.${NC}"
            echo -e "You can do this by adding the following line to your shell profile:"
            echo -e "  export PATH=\"$USER_BIN_DIR:\$PATH\""
        fi
    else
        echo -e "${RED}Error: CodeMap installation failed. Please try installing manually:${NC}"
        echo -e "$PYTHON_CMD -m pip install --user git+$REPO_URL"
        exit 1
    fi
}

show_final_instructions() {
    echo -e "\n${GREEN}=============================================${NC}"
    echo -e "${BOLD}CodeMap has been successfully installed!${NORMAL}"
    echo -e "${GREEN}=============================================${NC}"
    echo -e "\nUsage examples:"
    echo -e "  ${BOLD}codemap init${NORMAL}         - Initialize a new CodeMap project"
    echo -e "  ${BOLD}codemap generate${NORMAL}     - Generate documentation for your codebase"
    echo -e "  ${BOLD}codemap commit${NORMAL}       - Generate smart commit messages"
    echo -e "  ${BOLD}codemap pr create${NORMAL}    - Create a PR with AI-generated content"
    echo -e "\nFor more information, run: ${BOLD}codemap --help${NORMAL}"
    echo -e "See README.md for detailed usage instructions."
    echo -e "\nEnjoy using CodeMap!"
}

# Main installation process
print_header
check_python_version
check_pip
install_codemap
show_final_instructions 