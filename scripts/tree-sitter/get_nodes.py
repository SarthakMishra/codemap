"""Discover all tree-sitter repos with src/node-types.json files and download them locally."""

import json
import logging
import subprocess
from pathlib import Path

import requests
from github import Auth, Github
from github.GithubException import GithubException
from rich.logging import RichHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
rich_handler = RichHandler(level=logging.DEBUG, show_time=False, show_path=False, markup=True)
logger.addHandler(rich_handler)
logger.propagate = False

ORG = "tree-sitter"
REPO_PREFIX = "tree-sitter-"
NODE_TYPES_PATH = "src/node-types.json"
DOWNLOAD_DIR = Path("src/codemap/processor/tree_sitter/schema/languages/json")


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


def discover_node_types_files(token: str) -> dict:
	"""Discover node-types.json files for repos listed in sources.json. Return a mapping of language to raw URL."""
	auth = Auth.Token(token)
	g = Github(auth=auth)
	node_types_map = {}
	# Load sources.json
	sources_path = Path("src/codemap/processor/tree_sitter/schema/languages/sources.json")
	with sources_path.open() as f:
		sources = json.load(f)
	try:
		for lang, info in sources.items():
			repo_url = info["repo_url"]
			# Parse org and repo from URL
			parts = repo_url.rstrip("/").split("/")
			if len(parts) < 2:
				logger.warning(f"[yellow]Invalid repo_url for {lang}: {repo_url}[yellow]")
				continue
			org, repo_name = parts[-2], parts[-1]
			try:
				repo = g.get_repo(f"{org}/{repo_name}")
			except Exception as e:
				logger.warning(f"[yellow]Could not get repo {org}/{repo_name}: {e}[yellow]")
				continue
			logger.info(f"[cyan]Checking repo:[/cyan] {repo.full_name}")
			# Try src/node-types.json first
			try:
				repo.get_contents(NODE_TYPES_PATH)
				node_types_rel_path = NODE_TYPES_PATH
				base_url = f"https://raw.githubusercontent.com/{org}/{repo_name}/master/{node_types_rel_path}"
				node_types_map[lang] = {
					"repo": repo_name,
					"node_types_url": base_url,
				}
				logger.info(f"[green]Found node-types.json in {repo_name}/{node_types_rel_path}.[/green]")
				continue
			except GithubException as e:
				if e.status != 404:
					logger.warning(f"[yellow]Error accessing {repo_name}/{NODE_TYPES_PATH}: {e}[yellow]")
				# Fallback 1: try [language]/src/node-types.json
				fallback_path1 = f"{lang}/src/node-types.json"
				try:
					repo.get_contents(fallback_path1)
					node_types_rel_path = fallback_path1
					base_url = f"https://raw.githubusercontent.com/{org}/{repo_name}/master/{node_types_rel_path}"
					node_types_map[lang] = {
						"repo": repo_name,
						"node_types_url": base_url,
					}
					logger.info(f"[yellow]Found node-types.json in {repo_name}/{fallback_path1} (fallback 1).[/yellow]")
					continue
				except GithubException as e2:
					if e2.status != 404:
						logger.warning(f"[yellow]Error accessing {repo_name}/{fallback_path1}: {e2}[yellow]")
					# Fallback 2: try grammars/[language]/src/node-types.json
					fallback_path2 = f"grammars/{lang}/src/node-types.json"
					try:
						repo.get_contents(fallback_path2)
						node_types_rel_path = fallback_path2
						base_url = f"https://raw.githubusercontent.com/{org}/{repo_name}/master/{node_types_rel_path}"
						node_types_map[lang] = {
							"repo": repo_name,
							"node_types_url": base_url,
						}
						logger.info(
							f"[yellow]Found node-types.json in {repo_name}/{fallback_path2} (fallback 2).[/yellow]"
						)
						continue
					except GithubException as e3:
						if e3.status != 404:
							logger.warning(f"[yellow]Error accessing {repo_name}/{fallback_path2}: {e3}[yellow]")
						# Generalized fallback: search for all node-types.json files in the repo
						try:
							tree = repo.get_git_tree("HEAD", recursive=True)
							found_any = False
							for element in tree.tree:
								if element.type == "blob" and element.path.endswith("node-types.json"):
									parts = element.path.split("/")
									if len(parts) < 2:
										continue
									parent = parts[-2]
									base_url = (
										f"https://raw.githubusercontent.com/{org}/{repo_name}/master/{element.path}"
									)
									node_types_map[parent] = {
										"repo": repo_name,
										"node_types_url": base_url,
									}
									found_any = True
									logger.info(
										f"[blue]Found node-types.json in {repo_name}/{element.path} (general search, will save as {parent}.json).[/blue]"
									)
							if not found_any:
								logger.info(f"[yellow]No node-types.json found in {repo_name} by any method.[/yellow]")
						except Exception as e4:
							logger.warning(
								f"[yellow]Error searching tree for node-types.json in {repo_name}: {e4}[yellow]"
							)
	except Exception:
		logger.exception("Error discovering node-types.json files.")
	finally:
		g.close()
	return node_types_map


def download_node_types_file(language: str, node_types_url: str, dest_path: Path) -> bool:
	"""Download a node-types.json file and save it to dest_path."""
	try:
		logger.info(f"[bold cyan]Downloading[/bold cyan] {language}.json from {node_types_url}")
		response = requests.get(node_types_url, timeout=10)
		if response.status_code == 200:
			dest_path.parent.mkdir(parents=True, exist_ok=True)
			dest_path.write_text(response.text, encoding="utf-8")
			logger.info(f"[green]Saved[/green] to {dest_path}")
			return True
		logger.warning(f"[yellow]Failed to download[/yellow] {node_types_url} (status {response.status_code})")
		return False
	except Exception:
		logger.exception(f"Error downloading {node_types_url}")
		return False


def main():
	"""Main function to discover and download node-types.json files."""
	token = get_github_token_from_gh_cli()
	if not token:
		logger.error("[red]No GitHub token available. Exiting.[/red]")
		return
	node_types_map = discover_node_types_files(token)
	if not node_types_map:
		logger.warning("[yellow]No node-types.json files discovered. Exiting.[/yellow]")
		return
	for language, info in node_types_map.items():
		node_types_url = info.get("node_types_url")
		if not node_types_url:
			logger.warning(f"[yellow]No node_types_url for {language}, skipping.[/yellow]")
			continue
		dest_path = DOWNLOAD_DIR / f"{language}.json"
		download_node_types_file(language, node_types_url, dest_path)


if __name__ == "__main__":
	logger.info("[bold]Discovering and syncing node-types.json files...[/bold]")
	main()
	logger.info("[bold green]Done syncing node-types.json files.[/bold green]")
