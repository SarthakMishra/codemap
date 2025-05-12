"""Discover all tree-sitter repos with queries/*.scm files and return a mapping."""

import json
import logging
import subprocess
from pathlib import Path

from github import Auth, Github
from github.GithubException import GithubException
from rich.logging import RichHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
rich_handler = RichHandler(level=logging.DEBUG, show_time=False, show_path=False, markup=True)
logger.addHandler(rich_handler)
logger.propagate = False

SCM_JSON_PATH = Path("src/codemap/processor/tree_sitter/scm/ts_grammar_map.json")
ORG = "tree-sitter"
REPO_PREFIX = "tree-sitter-"
QUERIES_DIR = "queries"


def get_github_token_from_gh_cli() -> str:
	"""Get the GitHub token from the gh CLI tool."""
	try:
		result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True)
		token = result.stdout.strip()
		if not token:
			logger.error("[red]No token returned from gh CLI.[/red]")
			return ""
		logger.info("[green]GitHub token obtained from gh CLI.[/green]")
		return token
	except Exception:
		logger.exception("Failed to get GitHub token from gh CLI.")
		return ""


def discover_tree_sitter_scm(token: str) -> dict:
	"""Discover all tree-sitter repos with queries/*.scm files and return a mapping."""
	auth = Auth.Token(token)
	g = Github(auth=auth)
	scm_map = {}
	try:
		org = g.get_organization(ORG)
		repos = org.get_repos()
		for repo in repos:
			if not repo.name.startswith(REPO_PREFIX):
				continue
			logger.info(f"[cyan]Checking repo:[/cyan] {repo.name}")
			try:
				contents = repo.get_contents(QUERIES_DIR)
			except GithubException as e:
				if e.status == 404:
					logger.info(f"[yellow]No 'queries' dir in {repo.name}.[/yellow]")
					continue
				logger.warning(f"[yellow]Error accessing {repo.name}/queries: {e}[yellow]")
				continue
			scm_files = [f.name for f in contents if f.type == "file" and f.name.endswith(".scm")]
			if not scm_files:
				logger.info(f"[yellow]No .scm files in {repo.name}/queries.[/yellow]")
				continue
			base_url = f"https://raw.githubusercontent.com/{ORG}/{repo.name}/master/queries/"
			lang = repo.name.removeprefix(REPO_PREFIX)
			scm_map[lang] = {
				"repo": repo.name,
				"files": scm_files,
				"base_url": base_url,
			}
			logger.info(f"[green]Found {len(scm_files)} .scm files in {repo.name}/queries.[/green]")
	except Exception:
		logger.exception("Error discovering tree-sitter SCM files.")
	finally:
		g.close()
	return scm_map


def update_scm_json(scm_map: dict):
	"""Update the ts_grammar_map.json file with the new mapping."""
	try:
		SCM_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
		with SCM_JSON_PATH.open("w", encoding="utf-8") as f:
			json.dump(scm_map, f, indent=2)
		logger.info(f"[bold green]Updated {SCM_JSON_PATH} with {len(scm_map)} languages.[/bold green]")
	except Exception:
		logger.exception(f"Failed to update {SCM_JSON_PATH}")


def main():
	"""Main function to discover and update the SCM JSON file."""
	token = get_github_token_from_gh_cli()
	if not token:
		logger.error("[red]No GitHub token available. Exiting.[/red]")
		return
	scm_map = discover_tree_sitter_scm(token)
	if not scm_map:
		logger.warning("[yellow]No SCM files discovered. Exiting.[/yellow]")
		return
	update_scm_json(scm_map)


if __name__ == "__main__":
	main()
