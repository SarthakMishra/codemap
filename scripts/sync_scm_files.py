"""Download and update local SCM Query files from respective repositories."""

import json
import logging
from pathlib import Path

import requests
from rich.logging import RichHandler

# --- Configuration ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
rich_handler = RichHandler(level=logging.DEBUG, show_time=False, show_path=False, markup=True)
logger.addHandler(rich_handler)
logger.propagate = False

SCM_ROOT = Path("src/codemap/processor/tree_sitter/scm")
TS_GRAMMAR_MAP_PATH = SCM_ROOT / "ts_grammar_map.json"


def load_language_scm_map() -> dict:
	"""Load the language-to-SCM map from ts_grammar_map.json."""
	if not TS_GRAMMAR_MAP_PATH.exists():
		logger.error(f"[red]Grammar map file not found:[/red] {TS_GRAMMAR_MAP_PATH}")
		return {}
	try:
		with TS_GRAMMAR_MAP_PATH.open("r", encoding="utf-8") as f:
			data = json.load(f)
		logger.info(f"Loaded language SCM map from {TS_GRAMMAR_MAP_PATH}")
		return data
	except Exception:
		logger.exception(f"Error loading {TS_GRAMMAR_MAP_PATH}")
		return {}


def download_scm_file(language: str, file_name: str, url: str, dest_path: Path) -> bool:
	"""Download a single SCM file and save it to dest_path."""
	try:
		logger.info(f"[bold cyan]Downloading[/bold cyan] {language}/{file_name} from {url}")
		response = requests.get(url, timeout=10)
		if response.status_code == 200:
			dest_path.parent.mkdir(parents=True, exist_ok=True)
			dest_path.write_text(response.text, encoding="utf-8")
			logger.info(f"[green]Saved[/green] to {dest_path}")
			return True
		logger.warning(f"[yellow]Failed to download[/yellow] {url} (status {response.status_code})")
		return False
	except Exception:
		logger.exception(f"Error downloading {url}")
		return False


def sync_scm_files():
	"""Download all SCM files for all configured languages."""
	language_scm_map = load_language_scm_map()
	for language, info in language_scm_map.items():
		base_url = info.get("base_url")
		files = info.get("files", [])
		for file_name in files:
			url = f"{base_url}{file_name}"
			dest_path = SCM_ROOT / language / file_name
			download_scm_file(language, file_name, url, dest_path)


if __name__ == "__main__":
	logger.info("[bold]Syncing Tree-sitter SCM query files...[/bold]")
	sync_scm_files()
	logger.info("[bold green]Done syncing SCM files.[/bold green]")
