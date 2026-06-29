"""Headless Firefox browsing via Playwright for LLM-readable web context."""

import logging
import re
from typing import Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

_browser_instance: Optional["HeadlessBrowser"] = None


class HeadlessBrowser:
    """Persistent headless Firefox session for search and page reads."""

    def __init__(self, settings: dict):
        self._settings = settings
        self._headless = settings.get("browse_headless", True)
        self._timeout_ms = settings.get("browse_timeout_ms", 25000)
        self._max_chars = settings.get("browse_max_text_chars", 6000)
        self._playwright = None
        self._browser = None
        self._page = None
        self._started = False

    def ensure_started(self) -> None:
        if self._started:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is not installed. Run: pip install playwright "
                "&& playwright install firefox"
            ) from exc

        logger.info("[Browse] Starting headless Firefox...")
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.firefox.launch(headless=self._headless)
        self._page = self._browser.new_page()
        self._page.set_default_timeout(self._timeout_ms)
        self._started = True
        logger.info("[Browse] Headless Firefox ready.")

    def close(self) -> None:
        try:
            if self._page:
                self._page.close()
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception as exc:
            logger.warning("[Browse] Close error: %s", exc)
        finally:
            self._page = None
            self._browser = None
            self._playwright = None
            self._started = False

    def search(self, query: str) -> str:
        """Search DuckDuckGo HTML and return condensed result text for the LLM."""
        return self.search_engine("duckduckgo", query)

    def search_engine(self, engine: str, query: str) -> str:
        """Search a specific engine and return condensed result text."""
        self.ensure_started()
        engine = engine.lower().strip()
        if engine == "google":
            url = f"https://www.google.com/search?q={quote_plus(query)}&hl=en"
        elif engine == "bing":
            url = f"https://www.bing.com/search?q={quote_plus(query)}"
        elif engine == "yahoo":
            url = f"https://search.yahoo.com/search?p={quote_plus(query)}"
        elif engine == "duckduckgo":
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        else:
            raise ValueError(f"Unknown search engine: {engine}")

        logger.info("[Browse] %s search: %s", engine, query[:80])
        self._page.goto(url, wait_until="domcontentloaded")
        wait_ms = self._settings.get("layer0_search_wait_ms", 1500)
        if engine == "google":
            wait_ms = max(wait_ms, self._settings.get("google_ai_wait_ms", 2000))
        self._page.wait_for_timeout(wait_ms)
        html = self._page.content()
        return self._format_engine_results(html, engine, query)

    def google_structured_search(self, wrapped_query: str) -> str:
        """Google Search — extract page text (AI Overview often includes [start]...[end])."""
        self.ensure_started()
        url = f"https://www.google.com/search?q={quote_plus(wrapped_query)}&hl=en"
        logger.info("[Browse] Google AI query (%d chars)", len(wrapped_query))
        self._page.goto(url, wait_until="domcontentloaded")
        wait_ms = self._settings.get("google_ai_wait_ms", 3500)
        self._page.wait_for_timeout(wait_ms)
        text = self._read_page_text()
        # Prefer AI Overview region when present
        for selector in (
            "[data-container-id='aimicrophone']",
            "div[data-attrid='wa:/description']",
            "div.Y3BBE",
            "div.WaaZC",
        ):
            try:
                loc = self._page.locator(selector).first
                if loc.count() > 0:
                    block = loc.inner_text(timeout=2000)
                    if block and len(block) > 20:
                        return self._trim_text(block)
            except Exception:
                continue
        return text

    def open_url(self, url: str) -> str:
        """Navigate to a URL and return readable page text."""
        self.ensure_started()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        logger.info("[Browse] Open: %s", url)
        self._page.goto(url, wait_until="domcontentloaded")
        return self._read_page_text()

    def read_page(self) -> str:
        """Read the current page as plain text."""
        self.ensure_started()
        if not self._page.url or self._page.url == "about:blank":
            return "No page loaded."
        return self._read_page_text()

    def _read_page_text(self) -> str:
        assert self._page is not None
        try:
            text = self._page.inner_text("body")
        except Exception:
            text = self._page.content()
        return self._trim_text(text)

    def _format_engine_results(self, html: str, engine: str, query: str) -> str:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        lines = [f"Web search results ({engine}) for: {query}", ""]
        blocks: list = []

        if engine == "google":
            blocks = (
                soup.select("div.g")
                or soup.select("div[data-sokoban-container]")
                or soup.select("div.tF2Cxc")
            )
        elif engine == "bing":
            blocks = soup.select("li.b_algo") or soup.select(".b_algo")
        elif engine == "yahoo":
            blocks = soup.select("div.algo") or soup.select(".dd.algo")
        elif engine == "duckduckgo":
            blocks = soup.select("div.result") or soup.select(".web-result")

        if blocks:
            for idx, block in enumerate(blocks[:8], start=1):
                title_el = (
                    block.select_one("h3")
                    or block.select_one(".result__a")
                    or block.select_one("a")
                )
                snippet_el = (
                    block.select_one(".VwiC3b")
                    or block.select_one(".b_caption p")
                    or block.select_one(".compText")
                    or block.select_one(".result__snippet")
                    or block.select_one("p")
                )
                title = title_el.get_text(" ", strip=True) if title_el else ""
                snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
                if title or snippet:
                    lines.append(f"{idx}. {title}")
                    if snippet:
                        lines.append(f"   {snippet}")
                    lines.append("")

        if len(lines) <= 2:
            body = soup.get_text("\n", strip=True)
            lines.append(self._trim_text(body))

        return self._trim_text("\n".join(lines))

    def _format_search_results(self, html: str, query: str) -> str:
        return self._format_engine_results(html, "duckduckgo", query)

    def _trim_text(self, text: str) -> str:
        cleaned = re.sub(r"\n{3,}", "\n\n", text.strip())
        if len(cleaned) > self._max_chars:
            cleaned = cleaned[: self._max_chars] + "\n...[truncated]"
        return cleaned


def get_browser(settings: dict) -> HeadlessBrowser:
    global _browser_instance
    if _browser_instance is None:
        _browser_instance = HeadlessBrowser(settings)
    return _browser_instance


def reset_browser() -> None:
    global _browser_instance
    if _browser_instance is not None:
        _browser_instance.close()
        _browser_instance = None


def browse_search(query: str, settings: dict | None = None) -> str:
    from src.core.settings_loader import load_settings

    s = settings or load_settings()
    browser = get_browser(s)
    try:
        return browser.search(query)
    except Exception as exc:
        logger.exception("[Browse] search failed")
        return f"Search failed: {exc}"


def browse_open(url: str, settings: dict | None = None) -> str:
    from src.core.settings_loader import load_settings

    s = settings or load_settings()
    browser = get_browser(s)
    try:
        text = browser.open_url(url)
        preview = text[:800] + ("..." if len(text) > 800 else "")
        return f"Loaded {url}.\n\n{preview}"
    except Exception as exc:
        logger.exception("[Browse] open failed")
        return f"Could not open {url}: {exc}"


def browse_read(settings: dict | None = None) -> str:
    from src.core.settings_loader import load_settings

    s = settings or load_settings()
    browser = get_browser(s)
    try:
        return browser.read_page()
    except Exception as exc:
        logger.exception("[Browse] read failed")
        return f"Could not read page: {exc}"
