"""Lightweight HTTP search without a browser."""

import logging
import re
from urllib.parse import quote_plus

import requests

logger = logging.getLogger(__name__)


def ddg_lite_search(query: str, max_chars: int = 6000, timeout: int = 8) -> str:
    """DuckDuckGo HTML search via HTTP — no Playwright."""
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Jarvis3.0/1.0 (local personal assistant)"},
        )
        resp.raise_for_status()
        return _format_ddg_html(resp.text, query, max_chars)
    except Exception as exc:
        logger.warning("[SearchLite] DDG failed: %s", exc)
        return ""


def _format_ddg_html(html: str, query: str, max_chars: int) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    lines = [f"Web search results (duckduckgo-lite) for: {query}", ""]
    blocks = soup.select("div.result") or soup.select(".web-result")
    for idx, block in enumerate(blocks[:8], start=1):
        title_el = block.select_one(".result__a") or block.select_one("a")
        snippet_el = block.select_one(".result__snippet")
        title = title_el.get_text(" ", strip=True) if title_el else ""
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        if title or snippet:
            lines.append(f"{idx}. {title}")
            if snippet:
                lines.append(f"   {snippet}")
            lines.append("")
    if len(lines) <= 2:
        body = soup.get_text("\n", strip=True)
        lines.append(body)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text
