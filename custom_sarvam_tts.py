# custom_sarvam_tts.py — CORRECT for livekit-agents 1.4.2

from __future__ import annotations

import asyncio
import base64
import os
import logging
import numpy as np

from sarvamai import AsyncSarvamAI, AudioOutput, EventResponse

# ✅ CORRECT imports — confirmed from ElevenLabs plugin source in 1.4.x
from livekit.agents import APIConnectOptions, tts, utils
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

        if not self._api_key:
            raise ValueError("SARVAM_API_KEY is not set")
        if self._speaker not in VALID_SPEAKERS_V3:
            raise ValueError(
                f"Speaker '{self._speaker}' invalid for bulbul:v3. "
                f"Valid: {', '.join(sorted(VALID_SPEAKERS_V3))}"
            )

    # ✅ synthesize() for single-shot text
    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SarvamChunkedStream":
        return SarvamChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    # ✅ stream() for streaming mode — called by LiveKit agent pipeline
    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SarvamSynthStream":
        return SarvamSynthStream(
            tts=self,
            conn_options=conn_options,
        )


# ✅ NEW signature: _run(self, output_emitter) — required in livekit-agents 1.4.x
class SarvamChunkedStream(tts.ChunkedStream):
    """For single-shot synthesize() calls"""

    def __init__(self, *, tts: SarvamStreamingTTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts_ref = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        txt = self._input_text.strip()
        if not txt:
            return

        t      = self._tts_ref
        client = AsyncSarvamAI(api_subscription_key=t._api_key)

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=t._sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

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
                    pcm_bytes = base64.b64decode(message.data.audio)
                    output_emitter.push(pcm_bytes)
                elif isinstance(message, EventResponse):
                    if message.data.event_type == "final":
                        output_emitter.flush()
                        break


class SarvamSynthStream(tts.SynthesizeStream):
    """For streaming mode — used by the agent pipeline"""

    def __init__(self, *, tts: SarvamStreamingTTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts_ref = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        t      = self._tts_ref
        client = AsyncSarvamAI(api_subscription_key=t._api_key)

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=t._sample_rate,
            num_channels=1,
            stream=True,
            mime_type="audio/pcm",
        )

        # Collect streamed text tokens from the agent pipeline
        async for input_item in self._input_ch:
            # Each input_item is either a str token or FlushSentinel
            if isinstance(input_item, self._FlushSentinel):
                continue  # handled below per-segment

            txt = input_item.strip() if isinstance(input_item, str) else ""
            if not txt:
                continue

            segment_id = utils.shortuuid()
            output_emitter.start_segment(segment_id=segment_id)

            logger.debug(f"[SarvamTTS] Streaming segment: '{txt[:60]}'")

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

                chunk_count = 0
                async for message in ws:
                    if isinstance(message, AudioOutput):
                        pcm_bytes = base64.b64decode(message.data.audio)
                        output_emitter.push(pcm_bytes)
                        chunk_count += 1
                        if chunk_count == 1:
                            logger.debug("[SarvamTTS] ✅ First chunk delivered")
                    elif isinstance(message, EventResponse):
                        if message.data.event_type == "final":
                            output_emitter.end_input()
                            break
