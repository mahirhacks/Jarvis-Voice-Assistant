"""Google Did You Mean scraper (optional, off by default)."""

import logging
import random
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def google_dym(query: str, settings: dict) -> Optional[str]:
    if not settings.get("google_dym_enabled", False):
        return None
    try:
        delay = random.uniform(
            settings.get("google_dym_min_delay_seconds", 2),
            settings.get("google_dym_max_delay_seconds", 4),
        )
        time.sleep(delay)
        resp = requests.get(
            "https://www.google.com/search",
            params={"q": query},
            headers={"User-Agent": _USER_AGENT},
            timeout=settings.get("google_dym_timeout_seconds", 5),
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for anchor in soup.select("a.gL9Hya"):
            classes = anchor.get("class", [])
            if any("spell" in cls for cls in classes):
                text = anchor.get_text(strip=True)
                if text:
                    return text
        for anchor in soup.find_all("a"):
            anchor_text = anchor.get_text(strip=True)
            if anchor_text.lower().startswith("did you mean"):
                nested = anchor.find("i") or anchor.find("b") or anchor.find("em")
                if nested:
                    suggestion = nested.get_text(strip=True)
                    if suggestion:
                        return suggestion
        return None
    except Exception as exc:
        logger.info("[GoogleDYM] skipped for '%s': %s", query, exc)
        return None
