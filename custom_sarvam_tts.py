# custom_sarvam_tts.py — FINAL, matches livekit-agents 1.4.2 exactly

from __future__ import annotations

import asyncio
import base64
import os
import logging
import numpy as np
import io
import wave

from sarvamai import AsyncSarvamAI, AudioOutput, EventResponse

from livekit.agents import APIConnectOptions, tts, utils, tokenize
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger("sarvam-streaming-tts")
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")

VALID_SPEAKERS_V3 = {
    "shubh","ritu","rahul","pooja","simran","kavya","amit","ratan","rohan",
    "dev","ishita","shreya","manan","sumit","priya","aditya","kabir","neha",
    "varun","roopa","aayan","ashutosh","advait","amelia","sophia"
}

def _strip_wav_header(data: bytes) -> bytes:
    """Strip WAV header and return raw PCM bytes."""
    if data[:4] == b'RIFF':
        try:
            with wave.open(io.BytesIO(data)) as wf:
                return wf.readframes(wf.getnframes())
        except Exception:
            # If WAV parsing fails, try skipping standard 44-byte header
            return data[44:]
    return data  # already raw PCM, return as-is


class SarvamStreamingTTS(tts.TTS):
    def __init__(
        self,
        *,
        speaker: str = "kavya",
        language: str = "hi-IN",
        pace: float = 1.1,
        min_buffer_size: int = 50,
        sample_rate: int = 8000,
        api_key: str | None = None,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._speaker         = speaker
        self._language        = language
        self._pace            = pace
        self._min_buffer_size = min_buffer_size
        self._sample_rate     = sample_rate
        self._api_key         = api_key or SARVAM_API_KEY
        self._streams: set[SarvamSynthStream] = set()

        if not self._api_key:
            raise ValueError("SARVAM_API_KEY is not set")
        if self._speaker not in VALID_SPEAKERS_V3:
            raise ValueError(f"Speaker '{self._speaker}' invalid for bulbul:v3.")

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SarvamChunkedStream":
        return SarvamChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SarvamSynthStream":
        stream = SarvamSynthStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()


class SarvamChunkedStream(tts.ChunkedStream):
    """Used for session.say() and direct synthesize() calls"""

    def __init__(self, *, tts: SarvamStreamingTTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts_ref = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        t   = self._tts_ref
        txt = self._input_text.strip()
        if not txt:
            return

        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=t._sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

        client = AsyncSarvamAI(api_subscription_key=t._api_key)

        pushed_any = False

        async with client.text_to_speech_streaming.connect(model="bulbul:v3") as ws:
            await ws.configure(
                target_language_code=t._language,
                speaker=t._speaker,
                pace=t._pace,
                min_buffer_size=t._min_buffer_size,
                # DO NOT set output_audio_codec — let Sarvam return WAV
            )
            await ws.convert(txt)
            await ws.flush()

            async for message in ws:
                if isinstance(message, AudioOutput):
                    raw = base64.b64decode(message.data.audio)
                    pcm = _strip_wav_header(raw)
                    if pcm:
                        output_emitter.push(pcm)
                        pushed_any = True
                elif isinstance(message, EventResponse):
                    if message.data.event_type == "final":
                        output_emitter.flush()
                        break
        
        if not pushed_any:
            logger.error(f"[SarvamTTS] Sarvam returned ZERO audio bytes for text: '{txt[:80]}'")


class SarvamSynthStream(tts.SynthesizeStream):
    """Used by agent pipeline for streaming TTS during conversation"""

    def __init__(self, *, tts: SarvamStreamingTTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts_ref = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        t = self._tts_ref

        # FIRST thing — initialize the emitter
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=t._sample_rate,
            num_channels=1,
            stream=True,
            mime_type="audio/pcm",
        )

        # Start collecting tokens IMMEDIATELY into a buffer via background task
        collected: list[str] = []
        collection_done = asyncio.Event()

        async def _collect_input() -> None:
            async for data in self._input_ch:
                if isinstance(data, str):
                    collected.append(data)
            collection_done.set()

        collection_task = asyncio.create_task(_collect_input())

        # Wait for ALL tokens to arrive
        await collection_done.wait()
        await collection_task

        full_text = "".join(collected).strip()

        # Fallback: if input_ch was empty but framework passed text directly
        if not full_text and hasattr(self, '_input_text') and self._input_text:
            full_text = self._input_text.strip()
            logger.debug(f"[SarvamTTS] Recovered text from _input_text: '{full_text[:60]}'")

        if not full_text:
            logger.warning("[SarvamTTS] No text collected from input_ch — nothing to synthesize")
            return

        logger.debug(f"[SarvamTTS] Synthesising full response: '{full_text[:120]}'")

        # Step 2: ONE segment for the entire response
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        pushed_any = False
        try:
            client = AsyncSarvamAI(api_subscription_key=t._api_key)
            async with client.text_to_speech_streaming.connect(model="bulbul:v3") as ws:
                await ws.configure(
                    target_language_code=t._language,
                    speaker=t._speaker,
                    pace=t._pace,
                    min_buffer_size=t._min_buffer_size,
                    # DO NOT set output_audio_codec — let Sarvam return WAV
                )
                await ws.convert(full_text)
                await ws.flush()

                async for message in ws:
                    if isinstance(message, AudioOutput):
                        raw = base64.b64decode(message.data.audio)
                        pcm = _strip_wav_header(raw)
                        if pcm:
                            output_emitter.push(pcm)
                            pushed_any = True
                    elif isinstance(message, EventResponse):
                        if message.data.event_type == "final":
                            break
        finally:
            output_emitter.end_segment()

        if not pushed_any:
            logger.error(
                f"[SarvamTTS] Sarvam returned ZERO audio bytes for text: '{full_text[:80]}'"
            )
