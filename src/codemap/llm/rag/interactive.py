"""Interactive LLM interface for CodeMap."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich import print as rich_print
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
	from codemap.llm.client import UsageSummary

# Constants for UI thresholds
HIGH_SUCCESS_RATE_THRESHOLD = 95.0  # Percentage for green success rate
MEDIUM_SUCCESS_RATE_THRESHOLD = 80.0  # Percentage for yellow success rate
LOW_COST_THRESHOLD = 0.01  # USD threshold for green cost
MEDIUM_COST_THRESHOLD = 0.10  # USD threshold for yellow cost
VERY_LOW_COST_THRESHOLD = 0.001  # USD threshold for very cost-effective
HIGH_EFFICIENCY_THRESHOLD = 40.0  # Percentage for high response efficiency
LOW_EFFICIENCY_THRESHOLD = 10.0  # Percentage for low response efficiency
HEAVY_TOOL_USAGE_THRESHOLD = 5  # Number of tool calls for heavy usage
FAST_TOOL_DURATION = 1.0  # Seconds threshold for fast tool execution
FAST_RESPONSE_TIME = 5.0  # seconds for fast response
MEDIUM_RESPONSE_TIME = 15.0  # seconds for medium response
SLOW_RESPONSE_TIME = 30.0  # seconds for slow response
VERY_FAST_RESPONSE_TIME = 5.0  # seconds for very fast response


class RagUI:
	"""Interactive UI for the RAG process."""

	def __init__(self) -> None:
		"""Initialize the RAG UI."""
		self.console = Console()

	def format_ask_response(self, response_text: str | None) -> Markdown:
		"""
		Formats the AI's response text using Rich Markdown.

		Args:
			response_text (Optional[str]): The text response from the AI.

		Returns:
			Markdown: A Rich Markdown object ready for printing.

		"""
		if response_text is None:
			response_text = "*No response generated.*"
		# Basic Markdown formatting. Can be enhanced later to detect code blocks,
		# file paths, etc., and apply specific styling or links.
		return Markdown(response_text)

	def print_ask_result(self, result: dict[str, Any]) -> None:
		"""
		Prints the structured result of the ask command using Rich.

		Args:
			result (Dict[str, Any]): The structured result containing 'answer' and 'context'.

		"""
		answer = result.get("answer")
		context = result.get("context", [])

		# Print the main answer
		rich_print(Panel(self.format_ask_response(answer), title="[bold green]Answer[/]", border_style="green"))

		# Print the context used if there are any items
		if context:
			# Build a single string with all context items as a numbered list
			context_list = []
			for i, item in enumerate(context, 1):
				file_path = item.get("file_path", "Unknown")
				start_line = item.get("start_line", -1)
				end_line = item.get("end_line", -1)
				distance = item.get("distance", -1.0)

				# Create the list item text
				location = f"{file_path}"
				if start_line > 0 and end_line > 0:
					location += f" (lines {start_line}-{end_line})"

				# Format with relevance info
				relevance = f"(similarity: {1 - distance:.2f})" if distance >= 0 else ""
				list_item = f"[bold cyan]{i}.[/bold cyan] {location} [dim]{relevance}[/dim]"
				context_list.append(list_item)

			# Join all list items into a single string
			context_content = "\n".join(context_list)

			# Print a single panel with all context items
			rich_print(
				Panel(
					context_content, title="[bold yellow]Context Used[/]", border_style="yellow", title_align="center"
				)
			)

			rich_print()

	def format_content_for_context(self, context_items: list[dict[str, Any]]) -> str:
		"""
		Format context items into a string suitable for inclusion in prompts.

		Args:
			context_items: List of context dictionaries with file_path, start_line, end_line, and content

		Returns:
			Formatted string with code snippets and file information

		"""
		if not context_items:
			return "No relevant code found in the repository."

		formatted_parts = []

		for i, item in enumerate(context_items, 1):
			# Extract file information
			file_path = item.get("file_path", "Unknown file")
			start_line = item.get("start_line", -1)
			end_line = item.get("end_line", -1)
			content = item.get("content", "")

			# Create a header with file info
			header = f"[{i}] {file_path}"
			if start_line > 0 and end_line > 0:
				header += f" (lines {start_line}-{end_line})"

			# Format the code snippet with the header
			formatted_parts.append(f"{header}\n{'-' * len(header)}\n{content}\n")

		return "\n\n".join(formatted_parts)

	def print_usage_summary(self, usage_summary: UsageSummary) -> None:
		"""
		Print a comprehensive usage summary with beautiful formatting.

		Args:
			usage_summary: UsageSummary object with token usage, tool calls, and cost info
		"""
		# Create the main usage table with two columns for better space utilization
		usage_table = Table(show_header=True, header_style="bold magenta", border_style="bright_blue", box=None)
		usage_table.add_column("📊 Token & Request Metrics", style="cyan", no_wrap=True, min_width=25)
		usage_table.add_column("Value", style="green", justify="right", min_width=15)
		usage_table.add_column("", width=3)  # Separator column
		usage_table.add_column("⚡ Performance & Cost Metrics", style="cyan", no_wrap=True, min_width=25)
		usage_table.add_column("Value", style="green", justify="right", min_width=15)

		# Prepare left column data (Token & Request Metrics)
		left_metrics = [
			("📝 Request Tokens", f"{usage_summary.request_tokens:,}"),
			("💬 Response Tokens", f"{usage_summary.response_tokens:,}"),
			("📊 Total Tokens", f"[bold]{usage_summary.total_tokens:,}[/bold]"),
			("", ""),  # Separator
			("🔄 API Requests", str(usage_summary.requests)),
			("🔁 Iterations", str(usage_summary.iterations)),
		]

		# Add average tokens per request if we have requests
		if usage_summary.requests > 0:
			avg_tokens = usage_summary.total_tokens / usage_summary.requests
			left_metrics.append(("📈 Avg Tokens/Request", f"{avg_tokens:.0f}"))

		# Prepare right column data (Performance & Cost Metrics)
		right_metrics = []

		# Response time
		if usage_summary.response_time is not None:
			if usage_summary.response_time < FAST_RESPONSE_TIME:
				time_color = "green"
			elif usage_summary.response_time < MEDIUM_RESPONSE_TIME:
				time_color = "yellow"
			else:
				time_color = "red"
			time_text = f"[{time_color}]{usage_summary.response_time:.2f}s[/{time_color}]"
			right_metrics.append(("⏱️  Response Time", time_text))

		# Average time per request
		if usage_summary.response_time is not None and usage_summary.requests > 0:
			avg_time = usage_summary.response_time / usage_summary.requests
			avg_time_color = (
				"green" if avg_time < FAST_TOOL_DURATION else "yellow" if avg_time < MEDIUM_RESPONSE_TIME else "red"
			)
			right_metrics.append(("⚡ Avg Time/Request", f"[{avg_time_color}]{avg_time:.2f}s[/{avg_time_color}]"))

		right_metrics.append(("", ""))  # Separator

		# Tool usage summary with success rate
		tool_calls_value = str(usage_summary.total_tool_calls) if usage_summary.total_tool_calls > 0 else "[dim]0[/dim]"
		right_metrics.append(("🛠️  Total Tool Calls", tool_calls_value))

		# Calculate overall tool success rate
		if usage_summary.tool_calls:
			total_success = sum(tool.success_count for tool in usage_summary.tool_calls)
			total_calls = sum(tool.call_count for tool in usage_summary.tool_calls)
			if total_calls > 0:
				success_rate = (total_success / total_calls) * 100
				success_color = (
					"green"
					if success_rate >= HIGH_SUCCESS_RATE_THRESHOLD
					else "yellow"
					if success_rate >= MEDIUM_SUCCESS_RATE_THRESHOLD
					else "red"
				)
				right_metrics.append(
					("✅ Tool Success Rate", f"[{success_color}]{success_rate:.1f}%[/{success_color}]")
				)

		# Cost estimation with visual indicators
		if usage_summary.estimated_cost is not None:
			if usage_summary.estimated_cost < LOW_COST_THRESHOLD:
				cost_text = f"[green]${usage_summary.estimated_cost:.6f} USD[/green]"
			elif usage_summary.estimated_cost < MEDIUM_COST_THRESHOLD:
				cost_text = f"[yellow]${usage_summary.estimated_cost:.6f} USD[/yellow]"
			else:
				cost_text = f"[red]${usage_summary.estimated_cost:.6f} USD[/red]"
			right_metrics.append(("💰 Estimated Cost", cost_text))
		else:
			right_metrics.append(("💰 Estimated Cost", "[dim]Not available[/dim]"))

		# Fill shorter column with empty rows to align
		max_rows = max(len(left_metrics), len(right_metrics))
		while len(left_metrics) < max_rows:
			left_metrics.append(("", ""))
		while len(right_metrics) < max_rows:
			right_metrics.append(("", ""))

		# Add rows to table
		for i in range(max_rows):
			left_metric, left_value = left_metrics[i]
			right_metric, right_value = right_metrics[i]

			# Add visual separator
			separator = "│" if left_metric or right_metric else ""

			usage_table.add_row(left_metric, left_value, separator, right_metric, right_value)

		# Print main usage panel with enhanced styling
		rich_print(
			Panel(
				usage_table,
				title="[bold blue]📊 Usage Summary[/]",
				border_style="blue",
				title_align="center",
				padding=(1, 2),
			)
		)

		# If there are tool calls, show detailed tool breakdown
		if usage_summary.tool_calls:
			tool_table = Table(show_header=True, header_style="bold yellow", border_style="bright_green")
			tool_table.add_column("🛠️  Tool Name", style="cyan", min_width=20)
			tool_table.add_column("📞 Calls", style="green", justify="right", min_width=8)
			tool_table.add_column("✅ Success", style="green", justify="right", min_width=8)
			tool_table.add_column("❌ Errors", style="red", justify="right", min_width=8)
			tool_table.add_column("📈 Success Rate", style="blue", justify="right", min_width=12)
			tool_table.add_column("⏱️  Avg Duration", style="magenta", justify="right", min_width=12)

			for tool in usage_summary.tool_calls:
				# Calculate success rate for this tool
				success_rate = (tool.success_count / tool.call_count * 100) if tool.call_count > 0 else 0
				success_rate_color = (
					"green"
					if success_rate >= HIGH_SUCCESS_RATE_THRESHOLD
					else "yellow"
					if success_rate >= MEDIUM_SUCCESS_RATE_THRESHOLD
					else "red"
				)
				success_rate_text = f"[{success_rate_color}]{success_rate:.1f}%[/{success_rate_color}]"

				# Format duration
				avg_duration = f"{tool.total_duration / tool.call_count:.3f}s" if tool.call_count > 0 else "0.000s"
				duration_color = (
					"green" if tool.total_duration / max(tool.call_count, 1) < FAST_TOOL_DURATION else "yellow"
				)
				duration_text = f"[{duration_color}]{avg_duration}[/{duration_color}]"

				# Add error indicator to tool name if there were errors
				tool_name = tool.tool_name
				if tool.error_count > 0:
					tool_name = f"{tool.tool_name} [red]⚠️[/red]"

				tool_table.add_row(
					tool_name,
					str(tool.call_count),
					str(tool.success_count),
					str(tool.error_count) if tool.error_count > 0 else "[dim]0[/dim]",
					success_rate_text,
					duration_text,
				)

			rich_print(
				Panel(
					tool_table,
					title="[bold yellow]🛠️  Tool Usage Details[/]",
					border_style="yellow",
					title_align="center",
					padding=(1, 2),
				)
			)

		# Add a summary footer with key insights
		if usage_summary.total_tokens > 0:
			insights = []

			# Performance insights based on response time
			if usage_summary.response_time is not None:
				if usage_summary.response_time < FAST_RESPONSE_TIME:
					insights.append("🚀 Very fast response")
				elif usage_summary.response_time < MEDIUM_RESPONSE_TIME:
					insights.append("⚡ Fast response")
				elif usage_summary.response_time > SLOW_RESPONSE_TIME:
					insights.append("🐌 Slow response")

			# Cost insight
			if usage_summary.estimated_cost is not None:
				if usage_summary.estimated_cost < VERY_LOW_COST_THRESHOLD:
					insights.append("💚 Very cost-effective")
				elif usage_summary.estimated_cost > MEDIUM_COST_THRESHOLD:
					insights.append("💸 Higher cost usage")

			# Tool usage insight
			if usage_summary.total_tool_calls > HEAVY_TOOL_USAGE_THRESHOLD:
				insights.append("🔧 Heavy tool usage")
			elif usage_summary.total_tool_calls == 0:
				insights.append("📝 Text-only interaction")

			# Token efficiency insight
			if usage_summary.response_tokens > 0 and usage_summary.total_tokens > 0:
				response_ratio = (usage_summary.response_tokens / usage_summary.total_tokens) * 100
				if response_ratio > HIGH_EFFICIENCY_THRESHOLD:
					insights.append("🎯 High output efficiency")
				elif response_ratio < LOW_EFFICIENCY_THRESHOLD:
					insights.append("📤 Low output efficiency")

			if insights:
				insights_text = " • ".join(insights)
				rich_print(f"[dim italic]{insights_text}[/dim italic]")

		rich_print()  # Add spacing
