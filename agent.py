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
from custom_sarvam_tts import SarvamStreamingTTS
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
        **config
    }

def get_ist_time_context():
    """Returns a string indicating the current Date, Day, and Time in Indian Standard Time (IST)."""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    return f"\\n\\n[SYSTEM CONTEXT: Today is {now.strftime('%A, %B %d, %Y')}. The current time is {now.strftime('%I:%M %p')} IST. Use this for all relative date calculations.]"

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


    # ── Tool 2: Save Booking Intent ────────────────────────────────────────

    @llm.function_tool(
        description="Save the caller's intent to book an appointment for a specific date and time. Do this ONLY AFTER the caller has verbally confirmed the date, time, and their full name. This does not instantly book it, but queues it to be booked right after the call.",
    )
    async def save_booking_intent(
        self,
        start_time: Annotated[str, "The exact ISO 8601 start time with IST offset. Match this against the slots returned by check_availability. Example: '2026-02-24T10:00:00+05:30'"],
        caller_name: Annotated[str, "Full name of the caller as they stated it."],
        treatment_notes: Annotated[str, "Any relevant notes the caller mentioned — reason for appointment, service needed, etc."] = "",
    ) -> str:
        """Saves the booking details to memory so the agent can finalize it once they hang up."""
        logger.info(f"Booking intent saved: {start_time} for {caller_name}")
        
        # If the caller provides a name here, we update our context
        if caller_name and len(caller_name) > 1:
            self.caller_name = caller_name

        self.booking_intent = {
            "start_time": start_time,
            "caller_name": self.caller_name,
            "caller_phone": self.caller_phone,
            "notes": treatment_notes
        }
        
        return f"Successfully saved intent to book {start_time}. Tell the user their booking is confirmed and they will receive a text shortly."


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

        # Clear the booking intent
        self.booking_intent = None
        return (
            "No problem — I've cancelled your booking. "
            "Would you like to reschedule for another time?"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SARVAM-POWERED VOICE AGENT
# ══════════════════════════════════════════════════════════════════════════════

class OutboundAssistant(Agent):

    def __init__(self, agent_tools: AgentTools):
        tools = llm.find_function_tools(agent_tools)
        # Load the latest prompt from the UI dashboard config
        live_config = get_live_config()
        base_instructions = live_config.get("agent_instructions", "")
        
        # Inject the real-time IST clock into the prompt
        ist_context = get_ist_time_context()
        final_instructions = base_instructions + ist_context
        
        super().__init__(
            instructions=final_instructions,
            tools=tools,
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Say exactly this phrase without any extra thinking: 'Namaste! Welcome to Daisy's Med Spa. Main aapki kaise madad kar sakti hoon? I can answer questions about our treatments or help you book an appointment.'"
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
    


    # ── Build agent ───────────────────────────────────────────────────────
    agent = OutboundAssistant(agent_tools=agent_tools)

    # ── Read live configuration ───────────────────────────────────────────
    live_config = get_live_config()
    delay_setting = live_config.get("stt_min_endpointing_delay", 0.6)
    llm_model = live_config.get("llm_model", "gpt-4o-mini")
    tts_voice = live_config.get("tts_voice", "rohan")

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
            language="hi-IN",        # Force Hindi/Hinglish detection
            model="saaras:v3",
            mode="translate",        # Translate to enable faster English LLM processing
            flush_signal=True,
        ),
        llm=openai.LLM(
            model=llm_model,     # Dynamic LLM choice from dashboard
        ),
        tts=SarvamStreamingTTS(
            speaker=tts_voice,           # Dynamically set from UI config
            language="hi-IN",
            pace=1.1,
            min_buffer_size=50,
        ),
        turn_detection="stt",
        min_endpointing_delay=0.15,     # Raised from 0.07 — stops false self-interruptions
        allow_interruptions=True,
    )

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            close_on_disconnect=False,
        ),
    )

    FILLERS = [
        "Haan, ek second...",
        "Ji, dekhte hain...",
        "Bilkul, abhi batata hoon...",
    ]

    @session.on("user_speech_committed")
    def _on_speech_committed(ev):
        transcript = ev.user_transcript.strip()
        if len(transcript) > 3:
            filler = random.choice(FILLERS)
            asyncio.create_task(
                session.say(filler, add_to_chat_ctx=False)
            )

    logger.info("[AGENT] Session live — waiting for caller audio.")
    call_start_time = datetime.now()

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
                notify_booking_confirmed(
                    caller_name=intent["caller_name"],
                    caller_phone=intent["caller_phone"],
                    booking_time_iso=intent["start_time"],
                    booking_id=result.get("booking_id"),
                    notes=intent["notes"],
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
        
        from db import save_call_log
        save_call_log(
            phone=caller_phone,
            duration=duration,
            transcript=transcript_text,
            summary=booking_status_msg
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
