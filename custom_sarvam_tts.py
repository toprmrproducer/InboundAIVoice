from __future__ import annotations

import asyncio
import base64
import logging
import os

from sarvamai import AsyncSarvamAI, AudioOutput, EventResponse
from livekit.agents import APIConnectOptions, tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger("sarvam-streaming-tts")

VALID_SPEAKERS = {
    "shubh","ritu","rahul","pooja","simran","kavya","amit","ratan","rohan",
    "dev","ishita","shreya","manan","sumit","priya","aditya","kabir","neha",
    "varun","roopa","aayan","ashutosh","advait","amelia","sophia"
}


class SarvamStreamingTTS(tts.TTS):
    def __init__(
        self,
        *,
        speaker: str = "kavya",
        language: str = "hi-IN",
        pace: float = 1.1,
        min_buffer_size: int = 50,
        api_key: str | None = None,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=22050,
            num_channels=1,
        )
        self._speaker = speaker
        self._language = language
        self._pace = pace
        self._min_buffer_size = min_buffer_size
        self._api_key = api_key or os.environ.get("SARVAM_API_KEY", "")
        if not self._api_key:
            raise ValueError("SARVAM_API_KEY is not set")

    def synthesize(self, text: str, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        return SarvamChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(self, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        return SarvamSynthStream(tts=self, conn_options=conn_options)


async def _call_sarvam(t: SarvamStreamingTTS, text: str, output_emitter: tts.AudioEmitter):
    """Single shared function — synthesizes text and pushes to emitter."""
    text = text.strip()
    if not text:
        return

    logger.debug(f"[SarvamTTS] Synthesising: '{text[:120]}'")
    pushed_any = False

    output_emitter.start_segment(segment_id=utils.shortuuid())
    try:
        client = AsyncSarvamAI(api_subscription_key=t._api_key)
        async with client.text_to_speech_streaming.connect(model="bulbul:v3") as ws:
            await ws.configure(
                target_language_code=t._language,
                speaker=t._speaker,
                pace=t._pace,
                min_buffer_size=t._min_buffer_size,
                output_audio_codec="pcm",
            )
            await ws.convert(text)
            await ws.flush()

            async for message in ws:
                if isinstance(message, AudioOutput):
                    audio_bytes = base64.b64decode(message.data.audio)
                    if audio_bytes:
                        output_emitter.push(audio_bytes)
                        pushed_any = True
                elif isinstance(message, EventResponse):
                    if message.data.event_type == "final":
                        break
    finally:
        output_emitter.end_segment()

    if not pushed_any:
        logger.error(f"[SarvamTTS] Sarvam returned ZERO audio for: '{text[:80]}'")


class SarvamChunkedStream(tts.ChunkedStream):
    """For session.say() and direct synthesize() calls."""

    def __init__(self, *, tts, input_text, conn_options):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts_ref = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=22050,
            num_channels=1,
            mime_type="audio/pcm",
        )
        await _call_sarvam(self._tts_ref, self._input_text, output_emitter)


class SarvamSynthStream(tts.SynthesizeStream):
    """For live agent pipeline streaming."""

    def __init__(self, *, tts, conn_options):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts_ref = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=22050,
            num_channels=1,
            stream=True,
            mime_type="audio/pcm",
        )

        # Drain input_ch for tokens
        parts: list[str] = []
        async for data in self._input_ch:
            if isinstance(data, str):
                parts.append(data)

        text = "".join(parts).strip()

        # CRITICAL: fallback to _pushed_text if input_ch was empty
        # (1.4.2 bypasses input_ch and uses _pushed_text directly on retries)
        if not text:
            text = getattr(self, "_pushed_text", "").strip()

        if not text:
            logger.warning("[SarvamTTS] No text anywhere — skipping")
            return

        await _call_sarvam(self._tts_ref, text, output_emitter)
