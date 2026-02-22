# custom_sarvam_tts.py
# Sarvam Bulbul v3 — true WebSocket streaming TTS for LiveKit Agents
# Uses official sarvamai SDK. First audio chunk arrives in ~250ms.

import asyncio
import base64
import os
import logging
import numpy as np

from sarvamai import AsyncSarvamAI
from sarvamai.types import AudioOutput, EventResponse

from livekit import rtc
from livekit.agents import tts
from livekit.agents.tts import TTSCapabilities, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger("sarvam-streaming-tts")

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")

VALID_SPEAKERS_V3 = {
    "shubh","ritu","rahul","pooja","simran","kavya","amit","ratan","rohan",
    "dev","ishita","shreya","manan","sumit","priya","aditya","kabir","neha",
    "varun","roopa","aayan","ashutosh","advait","amelia","sophia"
}


class SarvamStreamingTTS(tts.TTS):
    """
    Sarvam Bulbul v3 — true streaming TTS via official sarvamai SDK.
    Bypasses the LiveKit plugin's batch behaviour.
    First audio chunk: ~250ms vs 2,600ms batch.
    """

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
            capabilities=TTSCapabilities(streaming=True),
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
                f"Speaker '{self._speaker}' invalid. "
                f"Valid: {', '.join(sorted(VALID_SPEAKERS_V3))}"
            )

    # ✅ conn_options accepted and passed through — this was the crash
    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SarvamSynthStream":
        return SarvamSynthStream(
            tts_instance=self,
            text=text,
            conn_options=conn_options,
        )

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SarvamSynthStream":
        return SarvamSynthStream(
            tts_instance=self,
            text="",
            conn_options=conn_options,
        )


class SarvamSynthStream(tts.SynthesizeStream):

    def __init__(
        self,
        *,
        tts_instance: SarvamStreamingTTS,
        text: str,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts_instance, conn_options=conn_options)  # ✅ pass to parent
        self._tts_instance = tts_instance
        self._text         = text

    async def _run(self) -> None:
        tts_ref = self._tts_instance
        text    = self._text.strip()

        if not text:
            # logger.debug("[SarvamTTS] Empty text — skipping synthesis")
            return

        logger.debug(f"[SarvamTTS] Synthesising ({len(text)} chars): '{text[:60]}...'")

        client = AsyncSarvamAI(api_subscription_key=tts_ref._api_key)

        try:
            async with client.text_to_speech_streaming.connect(
                model="bulbul:v3"
            ) as ws:

                # 1 — Configure session
                await ws.configure(
                    target_language_code=tts_ref._language,
                    speaker=tts_ref._speaker,
                    pace=tts_ref._pace,
                    min_buffer_size=tts_ref._min_buffer_size,
                    output_audio_codec="pcm",   # Raw PCM = zero decode overhead
                )

                # 2 — Send text and flush to start generation immediately
                await ws.convert(text)
                await ws.flush()

                logger.debug("[SarvamTTS] Text sent, waiting for first chunk...")

                # 3 — Stream chunks to LiveKit as they arrive
                chunk_count = 0
                async for message in ws:

                    if isinstance(message, AudioOutput):
                        pcm_bytes = base64.b64decode(message.data.audio)

                        # Convert raw PCM bytes → int16 numpy array
                        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16)
                        samples  = len(audio_np)

                        # Wrap in LiveKit AudioFrame and push downstream
                        frame = rtc.AudioFrame(
                            data=audio_np.tobytes(),
                            sample_rate=tts_ref._sample_rate,
                            num_channels=1,
                            samples_per_channel=samples,
                        )
                        self._event_ch.send_nowait(
                            tts.SynthesizedAudio(
                                request_id=self._request_id,
                                segment_id=self._segment_id,
                                frame=frame,
                            )
                        )
                        chunk_count += 1
                        if chunk_count == 1:
                            logger.debug("[SarvamTTS] ✅ First audio chunk delivered")

                    elif isinstance(message, EventResponse):
                        if message.data.event_type == "final":
                            logger.debug(
                                f"[SarvamTTS] Stream complete. "
                                f"Total chunks: {chunk_count}"
                            )
                            break

        except Exception as e:
            logger.error(f"[SarvamTTS] Streaming error: {e}", exc_info=True)
            raise
