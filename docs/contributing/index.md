# Development Setup

Before contributing, please read our [Code of Conduct](code-of-conduct.md) and [Contributing Guidelines](guidelines.md).

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/SarthakMishra/codemap.git
    cd codemap
    ```

2.  **Install Prerequisites:**
    *   **Task:** Follow the official installation guide: [https://taskfile.dev/installation/](https://taskfile.dev/installation/)
    *   **uv:** Follow the official installation guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)   
    *   **Python:** Ensure you have Python 3.12 or later installed.

3.  **Set up the Virtual Environment:**
    ```bash
    # Create a virtual environment using uv (creates .venv directory)
    uv venv

    # Activate the virtual environment
    # On Linux/macOS (bash/zsh):
    source .venv/bin/activate
    # On Windows (Command Prompt):
    # .venv\Scripts\activate.bat
    # On Windows (PowerShell):
    # .venv\Scripts\Activate.ps1
    ```

4.  **Install Dependencies:**
    Install project dependencies, including development tools, using `uv`:
    ```bash
    # Installs dependencies from pyproject.toml including the 'dev' group
    uv sync --dev
    ```

5.  **Verify Setup:**
    You can list available development tasks using Task:
    ```bash
    task -l
    ```
    To run all checks and tests (similar to CI):
    ```bash
    task ci
    ```

For detailed contribution guidelines, branching strategy, and coding standards, please refer to our [Contributing Guide](guidelines.md). 