"""Browser control tools."""

import logging
import time
import webbrowser

import pyautogui

logger = logging.getLogger(__name__)


def browser_new_tab() -> str:
    pyautogui.hotkey("ctrl", "t")
    return "Opened a new tab."


def browser_close_tab() -> str:
    pyautogui.hotkey("ctrl", "w")
    return "Closed the tab."


def browser_reopen_tab() -> str:
    pyautogui.hotkey("ctrl", "shift", "t")
    return "Reopened the last tab."


def browser_next_tab() -> str:
    pyautogui.hotkey("ctrl", "tab")
    return "Switched to next tab."


def browser_previous_tab() -> str:
    pyautogui.hotkey("ctrl", "shift", "tab")
    return "Switched to previous tab."


def browser_refresh() -> str:
    pyautogui.press("f5")
    return "Refreshed the page."


def browser_stop_loading() -> str:
    pyautogui.press("escape")
    return "Stopped loading."


def browser_back() -> str:
    pyautogui.hotkey("alt", "left")
    return "Went back."


def browser_forward() -> str:
    pyautogui.hotkey("alt", "right")
    return "Went forward."


def browser_scroll_down() -> str:
    pyautogui.scroll(-5)
    return "Scrolled down."


def browser_scroll_up() -> str:
    pyautogui.scroll(5)
    return "Scrolled up."


def browser_zoom_in() -> str:
    pyautogui.hotkey("ctrl", "+")
    return "Zoomed in."


def browser_zoom_out() -> str:
    pyautogui.hotkey("ctrl", "-")
    return "Zoomed out."


def browser_reset_zoom() -> str:
    pyautogui.hotkey("ctrl", "0")
    return "Reset zoom."


def browser_focus_address_bar() -> str:
    pyautogui.hotkey("ctrl", "l")
    return "Focused the address bar."


def browser_open_downloads() -> str:
    pyautogui.hotkey("ctrl", "j")
    return "Opened downloads."


def browser_open_history() -> str:
    pyautogui.hotkey("ctrl", "h")
    return "Opened history."


def browser_bookmark_page() -> str:
    pyautogui.hotkey("ctrl", "d")
    return "Bookmarked this page."


def browser_search(query: str) -> str:
    webbrowser.open(f"https://www.google.com/search?q={query}")
    return f"Searching for {query}."


def browser_go_to_website(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    webbrowser.open(url)
    return f"Opening {url}."


def browser_find_on_page(query: str) -> str:
    pyautogui.hotkey("ctrl", "f")
    time.sleep(0.3)
    pyautogui.typewrite(query, interval=0.02)
    return f"Searching for {query} on this page."


def open_url(url: str) -> str:
    webbrowser.open(url)
    return f"Opened {url}."
