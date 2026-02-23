import os
import json
import logging
import certifi
import pytz
import re
import random
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Fix for macOS SSL certificate verification
os.environ["SSL_CERT_FILE"] = certifi.where()

logging.getLogger("hpack").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from livekit import api
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
    llm,
)
from livekit.plugins import openai, sarvam, silero
from typing import Annotated

CONFIG_FILE = "config.json"

def get_live_config():
    """Reads the latest config.json to inject dynamic prompts and VAD tuning."""
    config = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read config.json, falling back: {e}")
            
    return {
        "agent_instructions": config.get("agent_instructions", ""),
        "stt_min_endpointing_delay": config.get("stt_min_endpointing_delay", 0.6),
        "llm_model": config.get("llm_model", "gpt-4o-mini"),
        "tts_voice": config.get("tts_voice", "kavya"),
        "tts_language": config.get("tts_language", "hi-IN"),
        **config
    }

def get_ist_time_context():
    """Returns current IST date/time AND the next 7 days so the agent
    can resolve 'this Thursday' / 'next Monday' to exact ISO dates."""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    today_str = now.strftime('%A, %B %d, %Y')
    time_str  = now.strftime('%I:%M %p')

    # Build a day-by-day lookup for the next 7 days
    from datetime import timedelta
    days_lines = []
    for i in range(7):
        day = now + timedelta(days=i)
        label = "Today" if i == 0 else ("Tomorrow" if i == 1 else day.strftime('%A'))
        days_lines.append(f"  {label}: {day.strftime('%A %d %B %Y')} → ISO {day.strftime('%Y-%m-%d')}")
    days_block = "\n".join(days_lines)

    return (
        f"\n\n[SYSTEM CONTEXT]\n"
        f"Current date & time: {today_str} at {time_str} IST\n"
        f"Use the table below to resolve ANY relative day reference (e.g. 'this Friday', 'next Monday', 'day after tomorrow') to the correct ISO date:\n"
        f"{days_block}\n"
        f"Always use the ISO date from this table when calling save_booking_intent. Appointments are in IST (+05:30).]"
    )

from calendar_tools import get_available_slots, create_booking, cancel_booking
from notify import (
    notify_booking_confirmed,
    notify_booking_cancelled,
    notify_call_no_booking,
    notify_agent_error,
)

load_dotenv()
logger = logging.getLogger("outbound-agent")
logging.basicConfig(level=logging.INFO)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL CONTEXT — All AI-callable functions
# ══════════════════════════════════════════════════════════════════════════════

class AgentTools(llm.ToolContext):

    def __init__(self, caller_phone: str, caller_name: str = ""):
        super().__init__(tools=[])
        self.caller_phone = caller_phone
        self.caller_name = caller_name
        self.booking_intent: dict | None = None  # Stores details for post-call booking

        # ── State tracked across the call ──────────────────────────────────
        self.sip_domain             = os.getenv("VOBIZ_SIP_DOMAIN")
        self.ctx_api                = None
        self.room_name              = None
        self._sip_identity          = None # Will be set in entrypoint if needed for transfer

    # ── Tool 1: Transfer to Human ──────────────────────────────────────────

    @llm.function_tool(
        description=(
            "Transfer this call to a human agent immediately. "
            "Use this if: the caller explicitly asks for a human, "
            "the caller is angry or frustrated, or the query is outside your scope."
        )
    )
    async def transfer_call(self):
        logger.info("[TOOL] transfer_call triggered")
        destination = os.getenv("DEFAULT_TRANSFER_NUMBER")
            
        if destination and self.sip_domain and "@" not in destination:
            clean_dest = destination.replace("tel:", "").replace("sip:", "")
            destination = f"sip:{clean_dest}@{self.sip_domain}"
        
        if destination and not destination.startswith("sip:"):
            destination = f"sip:{destination}"
            
        try:
            if self.ctx_api and self.room_name and destination and self._sip_identity:
                await self.ctx_api.sip.transfer_sip_participant(
                    api.TransferSIPParticipantRequest(
                        room_name=self.room_name,
                        participant_identity=self._sip_identity,
                        transfer_to=destination,
                        play_dialtone=False
                    )
                )
                return "Transfer initiated successfully."
            else:
                return "I'm having trouble transferring right now. Please hold on."
        except Exception as e:
            logger.error(f"Transfer failed: {e}")
            return "I'm having trouble transferring right now. Please hold on."


    # ── Tool 2: End Call (auto-hangup) ────────────────────────────────────

    @llm.function_tool(
        description=(
            "End the call and hang up. Use this ONLY when: "
            "(1) the caller explicitly says 'bye', 'goodbye', 'cut the call', or has confirmed their booking and the conversation is complete. "
            "(2) the caller says they don't need anything else. "
            "Say a short goodbye BEFORE calling this tool."
        )
    )
    async def end_call(self) -> str:
        logger.info("[TOOL] end_call triggered — hanging up.")
        try:
            if self.ctx_api and self.room_name and self._sip_identity:
                await self.ctx_api.sip.transfer_sip_participant(
                    api.TransferSIPParticipantRequest(
                        room_name=self.room_name,
                        participant_identity=self._sip_identity,
                        transfer_to="tel:+0",  # Sends REFER to empty destination = hangup
                        play_dialtone=False,
                    )
                )
            return "Call ended."
        except Exception as e:
            logger.warning(f"[end_call] Graceful hangup failed, forcing disconnect: {e}")
            return "Goodbye!"


    # ── Tool 3: Save Booking Intent ────────────────────────────────────────

    @llm.function_tool(
        description="Save the caller's intent to book an appointment for a specific date and time. Do this ONLY AFTER the caller has verbally confirmed the date, time, full name, and email address. This queues the booking to be confirmed right after the call.",
    )
    async def save_booking_intent(
        self,
        start_time: Annotated[str, "The exact ISO 8601 start time with IST offset. Example: '2026-02-24T10:00:00+05:30'"],
        caller_name: Annotated[str, "Full name of the caller as they stated it."],
        caller_email: Annotated[str, "Email address of the caller for booking confirmation."] = "",
        treatment_notes: Annotated[str, "Any relevant notes — service needed, preferences, etc."] = "",
    ) -> str:
        logger.info(f"Booking intent saved: {start_time} for {caller_name}")
        if caller_name and len(caller_name) > 1:
            self.caller_name = caller_name
        self.booking_intent = {
            "start_time": start_time,
            "caller_name": self.caller_name,
            "caller_phone": self.caller_phone,
            "caller_email": caller_email,
            "notes": treatment_notes,
        }
        return f"Booking intent saved for {start_time}. Tell the caller their appointment is confirmed and they'll receive a confirmation text shortly."


    # ── Tool 4: Cancel Appointment ─────────────────────────────────────────

    @llm.function_tool(
        description=(
            "Cancel the appointment that was booked during THIS call. "
            "Use this if the caller changes their mind after booking. "
            "Only works if a booking was already made in this session."
        )
    )
    async def cancel_appointment(
        self,
        reason: Annotated[str, "Reason for cancellation as stated by the caller."] = "Caller changed their mind",
    ):
        logger.info(f"[TOOL] cancel_appointment: reason={reason}")
        if not self.booking_intent:
            return "I don't have an active booking from this call to cancel."
        self.booking_intent = None
        return "No problem — I've cancelled your booking. Would you like to reschedule for another time?"


# ══════════════════════════════════════════════════════════════════════════════
# SARVAM-POWERED VOICE AGENT
# ══════════════════════════════════════════════════════════════════════════════

class OutboundAssistant(Agent):

    def __init__(self, agent_tools: AgentTools, first_line: str = ""):
        tools = llm.find_function_tools(agent_tools)
        self._first_line = first_line
        live_config = get_live_config()
        base_instructions = live_config.get("agent_instructions", "")
        ist_context = get_ist_time_context()
        final_instructions = base_instructions + ist_context
        super().__init__(
            instructions=final_instructions,
            tools=tools,
        )

    async def on_enter(self):
        greeting = self._first_line or (
            "Namaste! Welcome to Daisy's Med Spa. "
            "Main aapki kaise madad kar sakti hoon? "
            "I can answer questions about our treatments or help you book an appointment."
        )
        await self.session.generate_reply(
            instructions=f"Say exactly this phrase: '{greeting}'"
        )


# ══════════════════════════════════════════════════════════════════════════════
# JOB ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

async def entrypoint(ctx: JobContext):
    logger.info(f"[JOB] id={ctx.job.id}")
    logger.info(f"[JOB] raw metadata='{ctx.job.metadata}'")

    # ── Parse metadata ─────────────────────────────────────────────────────
    phone_number = None
    call_type    = "inbound"
    raw_meta     = ctx.job.metadata or ""
    caller_name  = "Unknown"

    if raw_meta.strip():
        try:
            meta = json.loads(raw_meta)
            phone_number = (
                meta.get("phone_number")
                or meta.get("to")
                or meta.get("destination")
            )
            caller_name = meta.get("name", caller_name)
            if phone_number:
                call_type = "outbound"
                logger.info(f"[CALL] Outbound → {phone_number}")
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"[METADATA] Parse error: {e} — treating as inbound")

    # ── Connect to LiveKit room ────────────────────────────────────────────
    await ctx.connect()
    logger.info(f"[ROOM] Connected: {ctx.room.name}")

    # ── Outbound: dial via Vobiz SIP trunk ────────────────────────────────
    if call_type == "outbound" and phone_number:
        try:
            lk_api = api.LiveKitAPI(
                url=os.environ["LIVEKIT_URL"],
                api_key=os.environ["LIVEKIT_API_KEY"],
                api_secret=os.environ["LIVEKIT_API_SECRET"],
            )
            
            sip_trunk_id = os.environ.get("OUTBOUND_TRUNK_ID", os.environ.get("SIP_TRUNK_ID", ""))
            
            await lk_api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=sip_trunk_id,
                    sip_call_to=phone_number,
                    participant_identity=f"sip_{phone_number.replace('+', '')}",
                    participant_name="Caller",
                    wait_until_answered=True,
                )
            )
            await lk_api.aclose()
            logger.info(f"[SIP] Outbound call dispatched to {phone_number} and answered.")
        except Exception as e:
            logger.error(f"[SIP] Dispatch failed: {e}")
            notify_agent_error(phone_number or "unknown", str(e))
            return

    # ── Instantiate tools ─────────────────────────────────────────────────
    caller_phone = phone_number or "unknown"
    participant_identity = (
        f"sip_{caller_phone.replace('+', '')}"
        if phone_number else "inbound_caller"
    )

    # Initialize tools with the caller's known phone number
    agent_tools = AgentTools(
        caller_phone=caller_phone,
        caller_name=caller_name,
    )
    agent_tools._sip_identity = participant_identity
    agent_tools.ctx_api = ctx.api
    agent_tools.room_name = ctx.room.name
    


    # ── Read live configuration ───────────────────────────────────────────
    live_config = get_live_config()
    delay_setting = live_config.get("stt_min_endpointing_delay", 0.15)
    llm_model = live_config.get("llm_model", "gpt-4o-mini")
    tts_voice = live_config.get("tts_voice", "rohan")
    tts_language = live_config.get("tts_language", "hi-IN")
    first_line = live_config.get("first_line", "")

    # ── Build agent ───────────────────────────────────────────────────────
    agent = OutboundAssistant(agent_tools=agent_tools, first_line=first_line)

    # Override OS environment variables if they are set in the UI dashboard
    for key in ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "OPENAI_API_KEY", "SARVAM_API_KEY", "CAL_API_KEY", "TELEGRAM_BOT_TOKEN", "SUPABASE_URL", "SUPABASE_KEY"]:
        val = live_config.get(key.lower(), "")
        if val:
            os.environ[key] = val

    # --- Interruption state tracking ---
    global agent_is_speaking
    agent_is_speaking = False

    async def on_user_speech_started(session):
        """Fires instantly when VAD detects user started speaking."""
        global agent_is_speaking
        if agent_is_speaking and session.current_speech:
            await session.current_speech.interrupt()
            logger.debug("[INTERRUPT] Cut off agent — user started speaking")

    def before_tts_cb(agent_response: str) -> str:
        """
        Returns only the FIRST sentence to TTS.
        Remaining sentences are queued as separate interruptible chunks.
        """
        sentences = re.split(r'(?<=[।.!?])\s+', agent_response.strip())
        return sentences[0] if sentences else agent_response

    # ── Start Sarvam-powered session ──────────────────────────────────────
    session = AgentSession(
        stt=sarvam.STT(
            language="hi-IN",
            model="saaras:v3",
            mode="translate",
            flush_signal=True,
        ),
        llm=openai.LLM(
            model=llm_model,
        ),
        tts=sarvam.TTS(
            target_language_code=tts_language,
            model="bulbul:v3",
            speaker=tts_voice,
        ),
        turn_detection="stt",
        min_endpointing_delay=0.07,
        allow_interruptions=True,
    )

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            close_on_disconnect=False,
        ),
    )

    logger.info("[AGENT] Session live — waiting for caller audio.")
    call_start_time = datetime.now()

    # ── Start call recording via LiveKit Egress ─────────────────────────────
    egress_id = None
    lk_api_for_egress = None
    try:
        lk_api_for_egress = api.LiveKitAPI(
            url=os.environ.get("LIVEKIT_URL", ""),
            api_key=os.environ.get("LIVEKIT_API_KEY", ""),
            api_secret=os.environ.get("LIVEKIT_API_SECRET", ""),
        )
        egress_info = await lk_api_for_egress.egress.start_room_composite_egress(
            api.RoomCompositeEgressRequest(
                room_name=ctx.room.name,
                audio_only=True,
                file=api.EncodedFileOutput(
                    filepath="/recordings/{room_name}-{time}.mp3",
                    file_type=api.EncodedFileType.MP3,
                    s3=api.S3Upload(
                        access_key=os.environ.get("AWS_ACCESS_KEY_ID", ""),
                        secret=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
                        bucket=os.environ.get("LIVEKIT_RECORDINGS_BUCKET", ""),
                        region=os.environ.get("AWS_REGION", "ap-south-1"),
                    ) if os.environ.get("LIVEKIT_RECORDINGS_BUCKET") else None,
                ),
            )
        )
        egress_id = egress_info.egress_id
        logger.info(f"[RECORDING] Egress started: {egress_id}")
    except Exception as e:
        logger.warning(f"[RECORDING] Could not start egress (no S3 configured?): {e}")
        if lk_api_for_egress:
            await lk_api_for_egress.aclose()

    @session.on("agent_speech_started")
    def _agent_speech_started(ev):
        global agent_is_speaking
        agent_is_speaking = True
        logger.debug("[STATE] Agent speaking: True")

    @session.on("agent_speech_finished")
    def _agent_speech_finished(ev):
        global agent_is_speaking
        agent_is_speaking = False
        logger.debug("[STATE] Agent speaking: False")

    FILLER_WORDS = {
        "okay.", "okay", "ok", "uh", "hmm", "hm", "yeah", "yes",
        "no", "um", "ah", "oh", "right", "sure", "fine", "good",
        "haan", "han", "theek", "theek hai", "accha", "ji", "ha",
    }

    @session.on("user_speech_committed")
    def on_user_speech_committed(ev):
        global agent_is_speaking

        transcript = ev.user_transcript.strip()
        transcript_lower = transcript.lower().rstrip(".")

        # Gate 1: Echo — ignore if agent is still speaking
        if agent_is_speaking:
            logger.debug(f"[FILTER-ECHO] Dropped: '{transcript}'")
            return

        # Gate 2: Empty transcript (confidence:0.0 Saaras v3 behaviour)
        if not transcript or len(transcript) < 3:
            logger.debug(f"[FILTER-EMPTY] Dropped empty transcript")
            return

        # Gate 3: Filler words
        if transcript_lower in FILLER_WORDS:
            logger.debug(f"[FILTER-FILLER] Dropped filler: '{transcript}'")
            return

        logger.info(f"[TRANSCRIPT] Passing to LLM: '{transcript}'")

    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant):
        logger.info(f"[HANGUP] Participant disconnected: {participant.identity}")
        # Set flag so transcript filter ignores any final flush
        global agent_is_speaking
        agent_is_speaking = False  # Clear any stuck state
        # Trigger graceful shutdown
        asyncio.create_task(unified_shutdown_hook(ctx))

    # ══════════════════════════════════════════════════════════════════════
    # POST-CALL SHUTDOWN HOOK
    # ══════════════════════════════════════════════════════════════════════

    async def unified_shutdown_hook(shutdown_ctx: JobContext):
        logger.info("Agent shutdown sequence started. Checking for pending bookings...")
        
        booking_status_msg = "No booking"
        if agent_tools.booking_intent:
            from calendar_tools import async_create_booking
            intent = agent_tools.booking_intent
            logger.info(f"Executing post-call booking intent for {intent['start_time']}")
            result = await async_create_booking(
                start_time=intent["start_time"],
                caller_name=intent["caller_name"] or "Unknown Caller",
                caller_phone=intent["caller_phone"],
                notes=intent["notes"],
            )
            if result.get("success"):
                # Build short AI summary from transcript
                short_summary = transcript_text[:300].strip() if 'transcript_text' in dir() else ""
                notify_booking_confirmed(
                    caller_name=intent["caller_name"],
                    caller_phone=intent["caller_phone"],
                    booking_time_iso=intent["start_time"],
                    booking_id=result.get("booking_id"),
                    notes=intent["notes"],
                    tts_voice=tts_voice,
                    ai_summary=short_summary,
                )
                logger.info("Post-call booking executed and notification sent.")
                booking_status_msg = f"Booking Confirmed: {result.get('booking_id')}"
            else:
                logger.error(f"Failed to execute post-call booking: {result.get('message')}")
                booking_status_msg = f"Booking Failed: {result.get('message')}"
        else:
            logger.info("[SHUTDOWN] No booking made — sending follow-up notification.")
            notify_call_no_booking(
                caller_name=agent_tools.caller_name,
                caller_phone=agent_tools.caller_phone,
                call_summary="Caller did not schedule an appointment during this call.",
                tts_voice=tts_voice,
                duration_seconds=int((datetime.now() - call_start_time).total_seconds()),
            )

        # Build Transcript & Save to Supabase
        duration = int((datetime.now() - call_start_time).total_seconds())
        transcript_text = ""
        
        try:
            messages = agent.chat_ctx.messages   
            if callable(messages):
                messages = messages()            

            transcript_lines = []
            for msg in messages:
                if getattr(msg, 'role', None) in ["user", "assistant"]:
                    content = getattr(msg, 'content', "")
                    if isinstance(content, list):
                        content = " ".join([str(c) for c in content if isinstance(c, str)])
                    transcript_lines.append(f"[{msg.role.upper()}] {content}")
            transcript_text = "\n".join(transcript_lines)
        except Exception as e:
            logger.error(f"[SHUTDOWN] Could not read chat history: {e}")
            transcript_text = "unavailable"
        
        # ── Stop Egress Recording ────────────────────────────────────────────────
        recording_url = ""
        if egress_id and lk_api_for_egress:
            try:
                stop_info = await lk_api_for_egress.egress.stop_egress(
                    api.StopEgressRequest(egress_id=egress_id)
                )
                # Try to extract the download URL from file results
                for f in (stop_info.file_results or []):
                    if f.download_url:
                        recording_url = f.download_url
                        break
                logger.info(f"[RECORDING] Egress stopped. URL: {recording_url or '(no URL yet — may still be processing)'}")
            except Exception as e:
                logger.warning(f"[RECORDING] Could not stop egress: {e}")
            finally:
                try:
                    await lk_api_for_egress.aclose()
                except Exception:
                    pass

        from db import save_call_log
        save_call_log(
            phone=caller_phone,
            duration=duration,
            transcript=transcript_text,
            summary=booking_status_msg,
            recording_url=recording_url,
        )

    ctx.add_shutdown_callback(unified_shutdown_hook)


# ══════════════════════════════════════════════════════════════════════════════
# WORKER ENTRY
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="outbound-caller" 
    ))
