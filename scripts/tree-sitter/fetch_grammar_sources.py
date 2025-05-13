"""Fetch the list of parser sources from the Tree-Sitter wiki page."""

import json
import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from github import Github
from rich.logging import RichHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
rich_handler = RichHandler(level=logging.INFO, show_time=False, show_path=False, markup=True)
logger.addHandler(rich_handler)
logger.propagate = False

WIKI_URL = "https://github.com/tree-sitter/tree-sitter/wiki/List-of-parsers"
OUTPUT_PATH = Path("src/codemap/processor/tree_sitter/schema/languages/json/sources.json")
GITHUB_API_URL = "https://api.github.com/repos/"


def get_github_token_from_gh_cli() -> str:
	"""Get the GitHub token from the gh CLI tool."""
	import subprocess

	try:
		result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True)
		token = result.stdout.strip()
		if not token:
			logger.warning("[yellow]No token returned from gh CLI.[/yellow]")
			return ""
		logger.info("[green]GitHub token obtained from gh CLI.[/green]")
		return token
	except Exception:
		logger.warning("[yellow]Failed to get GitHub token from gh CLI. Using unauthenticated requests.[/yellow]")
		return ""


def fetch_wiki_html():
	"""Fetch the wiki page and return the HTML."""
	logger.info(f"Fetching wiki page: {WIKI_URL}")
	resp = requests.get(WIKI_URL, timeout=10)
	resp.raise_for_status()
	return resp.text


def parse_parsers(html):
	"""Parse the HTML and extract parser names and repo URLs."""
	soup = BeautifulSoup(html, "html.parser")
	ul = soup.select_one("#wiki-body > div > ul")
	if not ul:
		logger.error("Could not find the main parser list <ul> in the wiki page.")
		return {}
	links = ul.find_all("a", href=True)
	parser_map = {}
	for link in links:
		href = link["href"]
		link.get_text(strip=True)
		if "github.com" in href and "tree-sitter-" in href:
			# Try to extract parser name from repo URL
			parts = href.split("/")
			if len(parts) < 2:
				continue
			repo_name = parts[-1]
			parser_name = repo_name.replace("tree-sitter-", "")
			# Some links may be to subdirs or files, skip those
			if not repo_name.startswith("tree-sitter-"):
				continue
			# Normalize parser name
			parser_name = parser_name.lower()
			if parser_name not in parser_map:
				parser_map[parser_name] = []
			parser_map[parser_name].append(
				{
					"repo_url": href,
					"repo_name": repo_name,
					"org": parts[-2] if len(parts) >= 2 else None,
				}
			)
	return parser_map


def get_repo_info(repo_url, gh=None):
	"""Get org and star count for a repo using PyGithub."""
	try:
		parts = repo_url.rstrip("/").split("/")
		org, repo = parts[-2], parts[-1]
		if not gh:
			gh = Github()  # unauthenticated, but still uses API
		repo_obj = gh.get_repo(f"{org}/{repo}")
		return {
			"org": org,
			"stars": repo_obj.stargazers_count,
			"repo": repo,
			"full_name": repo_obj.full_name,
			"url": repo_url,
		}
	except Exception as e:
		logger.warning(f"[yellow]Failed to get repo info for {repo_url}: {e}[/yellow]")
		return {
			"org": None,
			"stars": 0,
			"repo": None,
			"full_name": None,
			"url": repo_url,
		}


def resolve_duplicates(parser_map, gh=None):
	"""For each parser, pick the best repo (tree-sitter org preferred, else pick one at random)."""
	resolved = {}
	for parser, repos in parser_map.items():
		# Prefer org == 'tree-sitter'
		ts_repos = [r for r in repos if r["org"] == "tree-sitter"]
		best = ts_repos[0] if ts_repos else repos[0]  # If multiple, just pick the first
		resolved[parser] = best
	return resolved


def main():
	"""Main function."""
	token = get_github_token_from_gh_cli()
	gh = Github(token) if token else None
	html = fetch_wiki_html()
	parser_map = parse_parsers(html)
	logger.info(f"Found {len(parser_map)} unique parser names.")
	resolved = resolve_duplicates(parser_map, gh)
	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	with OUTPUT_PATH.open("w", encoding="utf-8") as f:
		json.dump(resolved, f, indent=2)
	logger.info(f"[green]Wrote sources to {OUTPUT_PATH}[/green]")


if __name__ == "__main__":
	main()
