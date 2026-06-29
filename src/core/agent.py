"""Jarvis 3.0 agent — hands-free voice loop with layered intent pipeline."""

import asyncio
import logging
import time
from typing import Any, Optional, Protocol

from src.core.confirmation import ask_confirmation
from src.core.intent_resolver import infer_intent_heuristic, intent_to_tool_route, meets_confidence
from src.core.layer_pipeline import (
    LayerPipeline,
    needs_layer2_llm,
    plan_to_direct_route,
    resolve_tool_for_plan,
)
from src.core.music_fast_route import vocabulary_route
from src.core.ollama_client import OllamaClient
from src.core.session import Session
from src.core.tool_registry import ToolRegistry
from src.core.tool_schemas import build_tool_schemas, narrow_tool_schemas
from src.core.transcript_normalize import is_wake_echo, normalize_transcript
from src.core.transcript_resolver import resolve_transcript
from src.voice import tts

logger = logging.getLogger(__name__)


class WakeProtocol(Protocol):
    def wait_for_wake_word(self) -> None: ...


class STTProtocol(Protocol):
    def record_and_transcribe(self) -> Optional[str]: ...


class JarvisAgent:
    def __init__(
        self,
        settings: dict,
        wake: WakeProtocol,
        stt: STTProtocol,
        ollama: OllamaClient,
        registry: ToolRegistry,
        session: Session,
        tool_schemas: list | None = None,
        layer_pipeline: LayerPipeline | None = None,
    ):
        self._settings = settings
        self._wake = wake
        self._stt = stt
        self._ollama = ollama
        self._registry = registry
        self._session = session
        self._layer_pipeline = layer_pipeline
        self._tool_schemas = tool_schemas or build_tool_schemas(settings)
        self._running = False
        self._last_transcript = ""

    async def run(self) -> None:
        self._running = True
        logger.info("[Agent] Hands-free loop started.")

        while self._running:
            try:
                await asyncio.to_thread(self._wake.wait_for_wake_word)

                if self._settings.get("wake_ack_enabled", True):
                    blocking_ack = self._settings.get("wake_ack_blocking", True)
                    tts.speak("Yes?", blocking=blocking_ack)
                    delay = self._settings.get("post_wake_delay_seconds", 0.25)
                    if delay > 0:
                        await asyncio.sleep(delay)

                await self._prepare_for_listen()

                t0 = time.time()
                transcript = await asyncio.to_thread(self._stt.record_and_transcribe)
                logger.info("[Agent] STT took %.2fs", time.time() - t0)

                if not transcript or not transcript.strip():
                    tts.speak("I didn't hear anything.", blocking=False)
                    continue

                self._last_transcript = transcript.strip()
                logger.info("[Agent] Transcript: '%s'", self._last_transcript)

                await self._handle_transcript(self._last_transcript)

            except Exception:
                logger.exception("[Agent] Loop error — retrying in 2s.")
                await asyncio.sleep(2)

    async def handle_transcript_dry(self, transcript: str) -> dict[str, Any]:
        """Resolve without executing tools (for testing)."""
        result = resolve_transcript(
            transcript, self._settings, self._layer_pipeline,
        )
        return {
            "transcript": result.transcript,
            "normalized": result.normalized,
            "path": result.path,
            "tool": result.route.get("tool_name") if result.route else None,
            "arguments": result.route.get("arguments") if result.route else None,
            "ignored": result.ignored,
            "message": result.message,
        }

    async def _handle_transcript(self, transcript: str) -> None:
        if is_wake_echo(transcript):
            logger.info("[Agent] Ignoring wake echo: '%s'", transcript)
            return

        normalized = normalize_transcript(transcript)
        if normalized != transcript.strip():
            logger.info("[Agent] Normalized: '%s' → '%s'", transcript, normalized)

        t0 = time.time()
        resolved = resolve_transcript(
            normalized, self._settings, self._layer_pipeline,
        )
        logger.info(
            "[Agent] Resolve took %.2fs path=%s tool=%s",
            time.time() - t0,
            resolved.path,
            resolved.route.get("tool_name") if resolved.route else None,
        )

        if resolved.ignored:
            return

        if resolved.path in ("fast", "vocab", "heuristic", "heuristic+context", "layer1-direct"):
            if resolved.route:
                await self._execute_route(resolved.route, normalized)
                return

        if resolved.path == "layer1-needs-llm" and self._layer_pipeline:
            handled = await self._run_layer2(resolved, normalized)
            if handled:
                return
            if resolved.route:
                await self._execute_route(resolved.route, normalized)
                return

        if resolved.path == "unresolved" and self._layer_pipeline:
            handled = await self._run_layer_pipeline_full(normalized)
            if handled:
                return

        tts.speak("I didn't understand that.")

    async def _run_layer2(self, resolved, transcript: str) -> bool:
        """Layer 2 LLM when layer1 plan needs argument disambiguation."""
        pipeline = self._layer_pipeline
        assert pipeline is not None and resolved.route

        from src.core.models import LayerPlan

        plan = LayerPlan(
            transcript=transcript,
            verb=resolved.plan_verb,
            phrase=resolved.plan_phrase or transcript,
            tool=resolved.route["tool_name"],
            confidence=resolved.plan_confidence or "medium",
        )
        primary_tool = resolve_tool_for_plan(plan, self._settings)
        narrow = narrow_tool_schemas(primary_tool, self._settings)
        llm_result = await asyncio.to_thread(pipeline.execute, plan, narrow)
        return await self._handle_llm_result(llm_result, transcript)

    async def _prepare_for_listen(self) -> None:
        """Pause background media so the mic is not flooded during capture."""
        if not self._settings.get("stt_pause_media_during_capture", True):
            return
        state = self._session.get_state()
        if not state.last_played_video:
            return
        try:
            from src.tools import media

            await asyncio.to_thread(media.media_play_pause)
            logger.info("[Agent] Paused media for command capture")
            await asyncio.sleep(0.15)
        except Exception:
            logger.warning("[Agent] Could not pause media before STT", exc_info=True)

    async def _run_layer_pipeline_full(self, transcript: str) -> bool:
        pipeline = self._layer_pipeline
        assert pipeline is not None
        progressive = self._settings.get("layer_progressive_tts", True)

        if progressive:
            tts.speak("One moment.", blocking=False)

        gather = await asyncio.to_thread(pipeline.gather, transcript)
        plan = await asyncio.to_thread(pipeline.find_intent, gather)

        if not plan or plan.verb == "none" or plan.tool == "none":
            hint = infer_intent_heuristic(transcript, gather.context)
            if hint and meets_confidence(hint, "medium"):
                route = intent_to_tool_route(hint, self._settings)
                if route:
                    await self._execute_route(route, transcript)
                    return True
            return False

        if progressive:
            tts.speak("Got it.", blocking=False)

        route_info = plan_to_direct_route(plan, self._settings)
        if route_info and not needs_layer2_llm(plan, self._settings):
            await self._execute_route(route_info, transcript)
            return True

        primary_tool = resolve_tool_for_plan(plan, self._settings)
        narrow = narrow_tool_schemas(primary_tool, self._settings)
        llm_result = await asyncio.to_thread(pipeline.execute, plan, narrow)
        if await self._handle_llm_result(llm_result, transcript):
            return True

        if route_info:
            await self._execute_route(route_info, transcript)
            return True
        return False

    async def _handle_llm_result(self, llm_result: dict[str, Any], transcript: str) -> bool:
        if "error" in llm_result:
            err_type = llm_result.get("error_type", "unknown")
            if err_type == "connection":
                tts.speak("The brain is offline. Please start Ollama.")
            elif err_type == "timeout":
                tts.speak("I'm having trouble thinking right now. Try again.")
            return err_type in ("connection", "timeout")

        tool_calls = llm_result.get("tool_calls", [])
        if tool_calls:
            for tc in tool_calls[: self._settings.get("llm_max_tool_rounds", 2)]:
                await self._execute_tool(
                    tc["name"],
                    tc.get("arguments", {}),
                    transcript,
                    needs_confirmation=self._registry.is_risky(tc["name"]),
                    confirmation_message=f"Are you sure you want to {tc['name'].replace('_', ' ')}?",
                )
            return True

        content = (llm_result.get("content") or "").strip()
        if content:
            tts.speak(content)
            return True

        return False

    async def _execute_route(self, route_info: dict, transcript: str) -> None:
        if route_info.get("needs_confirmation"):
            msg = route_info.get("confirmation_message", "Are you sure?")
            confirmed = await asyncio.to_thread(
                ask_confirmation, self._stt, msg,
                self._settings.get("confirmation_max_attempts", 2),
            )
            if not confirmed:
                tts.speak("Cancelled.")
                return

        await self._execute_tool(
            route_info["tool_name"],
            route_info.get("arguments", {}),
            transcript,
        )

    async def _execute_tool(
        self,
        tool_name: str,
        arguments: dict,
        transcript: str,
        needs_confirmation: bool = False,
        confirmation_message: str = "Are you sure?",
    ) -> None:
        if needs_confirmation and self._registry.is_risky(tool_name):
            confirmed = await asyncio.to_thread(
                ask_confirmation, self._stt, confirmation_message,
                self._settings.get("confirmation_max_attempts", 2),
            )
            if not confirmed:
                tts.speak("Cancelled.")
                return

        if tool_name == "search_youtube" and "raw_transcript" not in arguments:
            arguments = dict(arguments)
            arguments["raw_transcript"] = transcript

        t0 = time.time()
        result = await asyncio.to_thread(self._registry.execute, tool_name, arguments)
        logger.info("[Agent] Tool %s took %.2fs success=%s", tool_name, time.time() - t0, result.success)

        if result.message:
            tts.speak(result.message)

        if tool_name == "stop_jarvis":
            self._running = False
