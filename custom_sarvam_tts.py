# custom_sarvam_tts.py — FINAL, matches livekit-agents 1.4.2 exactly

from __future__ import annotations

import asyncio
import base64
import os
import logging
import numpy as np

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

        async with client.text_to_speech_streaming.connect(model="bulbul:v3") as ws:
            await ws.configure(
                target_language_code=t._language,
                speaker=t._speaker,
                pace=t._pace,
                min_buffer_size=t._min_buffer_size,
                output_audio_codec="pcm",
            )
            await ws.convert(txt)
            await ws.flush()

            async for message in ws:
                if isinstance(message, AudioOutput):
                    output_emitter.push(base64.b64decode(message.data.audio))
                elif isinstance(message, EventResponse):
                    if message.data.event_type == "final":
                        output_emitter.flush()
                        break


class SarvamSynthStream(tts.SynthesizeStream):
    """Used by agent pipeline for streaming TTS during conversation"""

    def __init__(self, *, tts: SarvamStreamingTTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts_ref = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        t = self._tts_ref

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=t._sample_rate,
            num_channels=1,
            stream=True,
            mime_type="audio/pcm",
        )

        # Buffer text tokens, synthesize per sentence boundary
        sentence_buffer: list[str] = []

        async def _flush_sentence(text: str) -> None:
            text = text.strip()
            if not text:
                return

            logger.debug(f"[SarvamTTS] Synthesising sentence: '{text[:80]}'")

            segment_id = utils.shortuuid()
            output_emitter.start_segment(segment_id=segment_id)  # ← one per full sentence

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
                            output_emitter.push(base64.b64decode(message.data.audio))
                        elif isinstance(message, EventResponse):
                            if message.data.event_type == "final":
                                break
            finally:
                output_emitter.end_segment()  # ← MUST end before next start_segment()

        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                # Flush whatever is left in the buffer
                if sentence_buffer:
                    await _flush_sentence("".join(sentence_buffer))
                    sentence_buffer.clear()
            elif isinstance(data, str):
                sentence_buffer.append(data)
                # Check for sentence-ending punctuation and flush eagerly
                joined = "".join(sentence_buffer)
                if joined.endswith((".", "?", "!", "।")):
                    await _flush_sentence(joined)
                    sentence_buffer.clear()

        # Final drain if anything remains
        if sentence_buffer:
            await _flush_sentence("".join(sentence_buffer))
