"""Ollama client with native tool calling."""

import json
import logging
import subprocess
import time
from typing import Any, Optional

import requests

from src.tools.apps import list_app_keys

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, settings: dict):
        self._settings = settings
        self._model = settings.get("llm_model", "qwen3.5:4b")
        self._base_url = settings.get("llm_base_url", "http://localhost:11434")
        self._temperature = settings.get("llm_temperature", 0.05)
        self._num_predict = settings.get("llm_num_predict", 200)
        self._num_ctx = settings.get("llm_num_ctx", 4096)
        self._keep_alive = settings.get("llm_keep_alive", "30m")
        self._timeout = settings.get("llm_timeout_seconds", 18)
        self._use_python = settings.get("ollama_use_python_client", True)
        self._restart_attempted = False

    @property
    def model(self) -> str:
        return self._model

    def check_model(self) -> bool:
        return self._check_model_available(self._model)

    def _check_model_available(self, model: str) -> bool:
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            for m in models:
                if m == model or m.startswith(model.split(":")[0]):
                    return True
            logger.warning("Model %s not found. Available: %s", model, models)
            return False
        except Exception as exc:
            logger.warning("Ollama check failed: %s", exc)
            return False

    def layer_ask(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Layer 1 — isolated one-shot prompt (no prior context)."""
        return self.layer_ask_json(system, user, max_tokens)

    def layer_ask_json(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Layer 1 — JSON structured one-shot."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        options = {
            "temperature": self._settings.get("llm_temperature", 0.05),
            "num_ctx": self._num_ctx,
            "num_predict": max_tokens or self._settings.get("llm_num_predict", 96),
        }
        timeout = self._settings.get("llm_timeout_seconds", 18)
        result = self._chat_with_model(
            self._model, messages, None, options, timeout, format="json", think=False,
        )
        if "error" not in result and (result.get("content") or "").strip():
            return result
        return self._http_chat(
            messages, None, options, timeout,
            endpoint="generate", model=self._model, format="json", think=False,
        )

    def layer_execute(
        self,
        brief: str,
        primary_tool: str,
        tools: list[dict],
    ) -> dict[str, Any]:
        """Layer 2 — fresh memory, tool calling with layer model."""
        from src.tools.apps import list_app_keys

        apps = ", ".join(list_app_keys(self._settings))
        system = (
            "You are Jarvis execution layer on Windows. Layer 1 already analyzed the user. "
            f"Your primary tool is: {primary_tool}. Call it with correct arguments. "
            "You may call a different tool only if the primary tool cannot work. "
            f"Allowed apps for open_app: {apps}. "
            "For music use search_youtube with the corrected song/artist as query. "
            "Output one tool call when an action is needed; otherwise one short spoken reply."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": brief},
        ]
        options = {
            "temperature": self._settings.get("llm_temperature", 0.05),
            "num_ctx": self._num_ctx,
            "num_predict": self._settings.get("llm_num_predict", 200),
        }
        timeout = self._settings.get("llm_timeout_seconds", 20)
        return self._chat_with_model(
            self._model, messages, tools, options, timeout,
        )

    def _chat_with_model(
        self,
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]],
        options: dict,
        timeout: int,
        think: Optional[bool] = None,
        format: Optional[str] = None,
    ) -> dict[str, Any]:
        if self._use_python:
            result = self._python_chat(
                messages, tools, options, timeout,
                model=model, format=format, think=think,
            )
            if "error" not in result:
                return result
        result = self._http_chat(
            messages, tools, options, timeout,
            endpoint="chat", model=model, format=format, think=think,
        )
        if "error" not in result:
            return result
        return self._http_chat(
            messages, tools, options, timeout,
            endpoint="generate", model=model, format=format, think=think,
        )

    def warm_up(self) -> bool:
        apps = ", ".join(list_app_keys(self._settings)[:8])
        result = self.chat(
            [{"role": "user", "content": f"Say ready. You can open apps like {apps}."}],
            tools=None,
            timeout=max(self._timeout, 15),
        )
        return "error" not in result

    def chat_with_tools(
        self,
        transcript: str,
        session_context: dict,
        tools: list[dict],
    ) -> dict[str, Any]:
        ctx = self._format_context(session_context)
        apps = ", ".join(list_app_keys(self._settings))
        system = (
            "You are Jarvis, a Windows voice assistant. Always pick exactly one tool when the user wants an action. "
            "Use open_app for opening programs and utilities (task_manager, explorer, settings, calculator, "
            f"chrome, vscode, notepad, etc.). Allowed app_key values: {apps}. "
            "Use search_youtube only for play/listen/music requests. "
            "Use media_* tools for pause, volume, next track. "
            "Use restart for reboot/restart laptop. Use shutdown for shut down. Use lock to lock PC. "
            "Use run_workflow for study_mode, play_music, etc. "
            "Keep spoken replies under one short sentence when not calling a tool."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{ctx}\nUser said: \"{transcript}\""},
        ]
        return self.chat(messages, tools=tools)

    def parse_intent(
        self,
        transcript: str,
        search_context: str,
        session_context: dict,
    ) -> dict[str, Any]:
        """Classify unclear speech using web search context. Returns {content} with structured JSON."""
        ctx = self._format_context(session_context)
        system = (
            "You classify voice commands for Jarvis on Windows. "
            "The user spoke something unclear; you are given web search results for context. "
            "Output ONLY valid JSON with exactly these keys:\n"
            '{"action":"<one>","objective":"<string>","confidence":"high|medium|low"}\n\n'
            "Allowed actions:\n"
            "- play: user wants a song/music/video (put 'Artist - Title' or 'Title by Artist' in objective)\n"
            "- open_url: open a specific website (full URL or domain in objective)\n"
            "- search_web: general web search (search terms in objective)\n"
            "- open_app: open a program (app name in objective, e.g. chrome, vscode, task manager)\n"
            "- run_workflow: run a workflow (study_mode, assignment_mode, bug_bounty_mode, play_music)\n"
            "- answer: user asked a question; put a short spoken answer in objective (under 25 words)\n"
            "- none: cannot determine a useful action\n\n"
            "Rules:\n"
            "- If search results show a song/track, use action play with corrected song and artist in objective.\n"
            "- Fix spelling/capitalization from search results (e.g. VideoClub not video club).\n"
            "- objective must be specific and actionable, not the raw transcript.\n"
            "- Output JSON only, no markdown, no extra text."
        )
        user = (
            f"{ctx}\n" if ctx else ""
        ) + (
            f'User said: "{transcript}"\n\n'
            f"Web search results:\n{search_context}\n\n"
            "Respond with JSON only."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        timeout = self._settings.get("intent_resolver_timeout_seconds", 20)
        num_predict = self._settings.get("intent_resolver_num_predict", 160)
        options = {
            "temperature": 0.05,
            "num_ctx": self._num_ctx,
            "num_predict": num_predict,
        }
        if self._use_python:
            result = self._python_chat(
                messages, None, options, timeout, format="json", think=False,
            )
            if "error" not in result and (result.get("content") or "").strip():
                return result
        result = self._http_chat(
            messages, None, options, timeout, endpoint="chat", format="json", think=False,
        )
        if "error" not in result and (result.get("content") or "").strip():
            return result
        # Fallback: /api/generate with JSON format (more reliable on some models)
        return self._http_chat(
            messages, None, options, timeout, endpoint="generate", format="json", think=False,
        )

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        timeout: Optional[int] = None,
    ) -> dict[str, Any]:
        t = timeout or self._timeout
        options = {
            "temperature": self._temperature,
            "num_ctx": self._num_ctx,
            "num_predict": self._num_predict,
        }

        if self._use_python:
            result = self._python_chat(messages, tools, options, t)
            if "error" not in result:
                return result
            if result.get("error_type") == "connection":
                self._maybe_restart()

        result = self._http_chat(messages, tools, options, t, endpoint="chat")
        if "error" not in result:
            return result
        if result.get("error_type") == "connection":
            self._maybe_restart()

        return self._http_chat(messages, tools, options, t, endpoint="generate")

    def _python_chat(
        self, messages: list[dict], tools: Optional[list], options: dict, timeout: int,
        model: Optional[str] = None, format: Optional[str] = None, think: Optional[bool] = None,
    ) -> dict[str, Any]:
        try:
            from ollama import Client
            client = Client(host=self._base_url, timeout=timeout)
            kwargs: dict[str, Any] = {
                "model": model or self._model,
                "messages": messages,
                "stream": False,
                "keep_alive": self._keep_alive,
                "options": options,
            }
            if tools:
                kwargs["tools"] = tools
            if format:
                kwargs["format"] = format
            if think is not None:
                kwargs["think"] = think
            t0 = time.time()
            response = client.chat(**kwargs)
            logger.info("[Ollama] chat took %.2fs", time.time() - t0)
            return self._normalize_response(response)
        except ImportError:
            return {"error": "ollama package missing", "error_type": "import"}
        except Exception as exc:
            err = str(exc).lower()
            if "connection" in err or "refused" in err:
                return {"error": str(exc), "error_type": "connection"}
            return {"error": str(exc), "error_type": "unknown"}

    def _http_chat(
        self, messages: list[dict], tools: Optional[list], options: dict,
        timeout: int, endpoint: str,
        model: Optional[str] = None, format: Optional[str] = None, think: Optional[bool] = None,
    ) -> dict[str, Any]:
        try:
            use_model = model or self._model
            if endpoint == "chat":
                url = f"{self._base_url}/api/chat"
                payload: dict[str, Any] = {
                    "model": use_model, "messages": messages, "stream": False,
                    "keep_alive": self._keep_alive, "options": options,
                }
                if tools:
                    payload["tools"] = tools
                if format:
                    payload["format"] = format
                if think is not None:
                    payload["think"] = think
            else:
                url = f"{self._base_url}/api/generate"
                prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                payload = {
                    "model": use_model, "prompt": prompt, "stream": False,
                    "keep_alive": self._keep_alive, "options": options,
                }
                if format:
                    payload["format"] = format
                if think is not None:
                    payload["think"] = think
            t0 = time.time()
            resp = requests.post(url, json=payload, timeout=timeout)
            logger.info("[Ollama] %s took %.2fs status=%d", endpoint, time.time() - t0, resp.status_code)
            resp.raise_for_status()
            data = resp.json()
            if endpoint == "chat":
                return self._normalize_response(data)
            return self._normalize_response(data)
        except requests.exceptions.ConnectionError:
            return {"error": "Connection refused", "error_type": "connection"}
        except requests.exceptions.Timeout:
            return {"error": "Timeout", "error_type": "timeout"}
        except Exception as exc:
            return {"error": str(exc), "error_type": "unknown"}

    @staticmethod
    def _normalize_response(response: dict) -> dict[str, Any]:
        msg = response.get("message", response)
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        thinking = msg.get("thinking", "") if isinstance(msg, dict) else ""
        if not content and thinking:
            content = thinking
        if not content and isinstance(response, dict):
            content = response.get("response", "") or response.get("thinking", "")
        tool_calls = msg.get("tool_calls", []) if isinstance(msg, dict) else []
        normalized_calls = []
        for tc in tool_calls or []:
            fn = tc.get("function", tc)
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            normalized_calls.append({"name": name, "arguments": args})
        return {"content": content, "tool_calls": normalized_calls}

    @staticmethod
    def _format_context(ctx: dict) -> str:
        if not ctx:
            return ""
        parts = []
        for key in ("last_user_request", "last_artist", "last_song", "last_search_query"):
            if ctx.get(key):
                parts.append(f'{key}: "{ctx[key]}"')
        return "Context: " + "; ".join(parts) if parts else ""

    def _maybe_restart(self) -> bool:
        if self._restart_attempted:
            return False
        self._restart_attempted = True
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)
            return True
        except Exception:
            return False
