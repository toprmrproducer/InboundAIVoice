import os
import logging
import requests
import httpx
from datetime import datetime

logger = logging.getLogger("calendar-tools")

CAL_BASE = "https://api.cal.com/v1"


def get_cal_creds() -> dict:
    return {
        "api_key":  os.environ.get("CAL_API_KEY", ""),
        "event_id": int(os.environ.get("CAL_EVENT_TYPE_ID", "0") or "0"),
    }


# ─── Cal.com: Get available slots ─────────────────────────────────────────────

def get_available_slots(date_str: str) -> list:
    """
    Fetch open slots for a given date from Cal.com OR Google Calendar,
    depending on which is configured.
    date_str: "YYYY-MM-DD"
    """
    # Try Google Calendar first if configured (#36)
    gcal_id = os.environ.get("GOOGLE_CALENDAR_ID", "")
    gcal_creds = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", "google_creds.json")
    if gcal_id and os.path.exists(gcal_creds):
        try:
            return _get_slots_gcal(date_str, gcal_id, gcal_creds)
        except Exception as e:
            logger.warning(f"[GCAL] Falling back to Cal.com: {e}")

    # Default: Cal.com
    return _get_slots_calcom(date_str)


def _get_slots_calcom(date_str: str) -> list:
    creds = get_cal_creds()
    try:
        resp = requests.get(
            f"{CAL_BASE}/slots",
            headers={"Content-Type": "application/json"},
            params={
                "apiKey":      creds["api_key"],
                "eventTypeId": creds["event_id"],
                "startTime":   f"{date_str}T00:00:00.000Z",
                "endTime":     f"{date_str}T23:59:59.000Z",
            },
            timeout=8,
        )
        resp.raise_for_status()
        raw_slots = resp.json().get("data", {}).get("slots", {}).get(date_str, [])
        slots = []
        for s in raw_slots:
            dt = datetime.fromisoformat(s["time"])
            slots.append({"time": s["time"], "label": dt.strftime("%-I:%M %p")})
        logger.info(f"[CAL] {len(slots)} slots for {date_str}")
        return slots
    except Exception as e:
        logger.error(f"[CAL] get_available_slots error: {e}")
        return []


def _get_slots_gcal(date_str: str, calendar_id: str, creds_file: str) -> list:
    """
    Fetch busy slots from Google Calendar and compute free windows (#36).
    Requires: google-api-python-client, google-auth
    """
    from googleapiclient.discovery import build
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_file(
        creds_file,
        scopes=["https://www.googleapis.com/auth/calendar.readonly"],
    )
    service = build("calendar", "v3", credentials=creds)

    start = f"{date_str}T00:00:00+05:30"
    end   = f"{date_str}T23:59:59+05:30"

    result = service.freebusy().query(body={
        "timeMin": start,
        "timeMax": end,
        "items":   [{"id": calendar_id}],
    }).execute()

    busy_slots = result.get("calendars", {}).get(calendar_id, {}).get("busy", [])

    # Generate free 30-min slots between 10:00 and 19:00 IST
    import pytz
    from datetime import timedelta
    ist = pytz.timezone("Asia/Kolkata")
    day_start = ist.localize(datetime.strptime(f"{date_str} 10:00", "%Y-%m-%d %H:%M"))
    day_end   = ist.localize(datetime.strptime(f"{date_str} 19:00", "%Y-%m-%d %H:%M"))

    busy_ranges = []
    for b in busy_slots:
        bs = datetime.fromisoformat(b["start"]).astimezone(ist)
        be = datetime.fromisoformat(b["end"]).astimezone(ist)
        busy_ranges.append((bs, be))

    free_slots = []
    slot = day_start
    while slot < day_end:
        slot_end = slot + timedelta(minutes=30)
        is_busy = any(bs <= slot < be for bs, be in busy_ranges)
        if not is_busy:
            free_slots.append({
                "time":  slot.isoformat(),
                "label": slot.strftime("%-I:%M %p"),
            })
        slot = slot_end

    logger.info(f"[GCAL] {len(free_slots)} free slots for {date_str}")
    return free_slots


# ─── Create a booking ──────────────────────────────────────────────────────────

def create_booking(
    start_time: str,
    caller_name: str,
    caller_phone: str,
    notes: str = "",
) -> dict:
    """Synchronous wrapper — calls async_create_booking."""
    import asyncio
    try:
        return asyncio.get_event_loop().run_until_complete(
            async_create_booking(start_time, caller_name, caller_phone, notes)
        )
    except RuntimeError:
        return asyncio.run(async_create_booking(start_time, caller_name, caller_phone, notes))


async def async_create_booking(
    start_time: str,
    caller_name: str,
    caller_phone: str,
    notes: str = "",
) -> dict:
    """
    Book a slot — uses Google Calendar if configured, else Cal.com v2.
    start_time: ISO 8601 with IST offset e.g. "2026-02-24T10:00:00+05:30"
    Returns: {"success": bool, "booking_id": str|None, "message": str}
    """
    gcal_id    = os.environ.get("GOOGLE_CALENDAR_ID", "")
    gcal_creds = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", "google_creds.json")

    if gcal_id and os.path.exists(gcal_creds):
        return await _create_booking_gcal(start_time, caller_name, caller_phone, notes, gcal_id, gcal_creds)

    return await _create_booking_calcom(start_time, caller_name, caller_phone, notes)


async def _create_booking_calcom(
    start_time: str, caller_name: str, caller_phone: str, notes: str
) -> dict:
    creds = get_cal_creds()
    payload = {
        "eventTypeId": creds["event_id"],
        "start": start_time,
        "attendee": {
            "name":        caller_name,
            "email":       f"{caller_phone.replace('+','').replace(' ','')}@voiceagent.placeholder",
            "phoneNumber": caller_phone,
            "timeZone":    "Asia/Kolkata",
            "language":    "en",
        },
        "bookingFieldsResponses": {
            "notes": notes or f"Booked via AI voice agent. Phone: {caller_phone}",
        },
    }
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                "https://api.cal.com/v2/bookings",
                headers={
                    "Authorization":  f"Bearer {creds['api_key']}",
                    "cal-api-version": "2024-08-13",
                    "Content-Type":   "application/json",
                },
                json=payload,
            )
            if resp.status_code not in (200, 201):
                logger.error(f"[CAL] Booking failed {resp.status_code}: {resp.text}")
                return {"success": False, "booking_id": None, "message": resp.text}
            uid = resp.json().get("data", {}).get("uid", "unknown")
            logger.info(f"[CAL] Booking created: uid={uid}")
            return {"success": True, "booking_id": uid, "message": "Booking confirmed"}
    except httpx.TimeoutException:
        return {"success": False, "booking_id": None, "message": "Booking timed out."}
    except Exception as e:
        logger.error(f"[CAL] Booking error: {e}")
        return {"success": False, "booking_id": None, "message": str(e)}


async def _create_booking_gcal(
    start_time: str,
    caller_name: str,
    caller_phone: str,
    notes: str,
    calendar_id: str,
    creds_file: str,
) -> dict:
    """Create a Google Calendar event (#36)."""
    try:
        from googleapiclient.discovery import build
        from google.oauth2 import service_account
        from datetime import timedelta

        creds = service_account.Credentials.from_service_account_file(
            creds_file,
            scopes=["https://www.googleapis.com/auth/calendar"],
        )
        service = build("calendar", "v3", credentials=creds)

        dt_start = datetime.fromisoformat(start_time)
        dt_end   = dt_start + timedelta(minutes=30)

        event = {
            "summary":     f"Appointment — {caller_name}",
            "description": f"Phone: {caller_phone}\nNotes: {notes}\nBooked via RapidX AI Voice Agent",
            "start":       {"dateTime": dt_start.isoformat(), "timeZone": "Asia/Kolkata"},
            "end":         {"dateTime": dt_end.isoformat(),   "timeZone": "Asia/Kolkata"},
            "attendees":   [{"displayName": caller_name, "comment": caller_phone}],
        }

        created = service.events().insert(calendarId=calendar_id, body=event).execute()
        event_id = created.get("id", "unknown")
        logger.info(f"[GCAL] Event created: id={event_id}")
        return {"success": True, "booking_id": event_id, "message": "Google Calendar event created"}
    except Exception as e:
        logger.error(f"[GCAL] Create booking failed: {e}")
        return {"success": False, "booking_id": None, "message": str(e)}


# ─── Cancel a booking ──────────────────────────────────────────────────────────

def cancel_booking(booking_id: str, reason: str = "Cancelled by caller") -> dict:
    """Cancel a Cal.com booking by UID."""
    creds = get_cal_creds()
    try:
        resp = requests.delete(
            f"{CAL_BASE}/bookings/{booking_id}/cancel?apiKey={creds['api_key']}",
            headers={"Content-Type": "application/json"},
            json={"reason": reason},
            timeout=8,
        )
        resp.raise_for_status()
        logger.info(f"[CAL] Booking cancelled: {booking_id}")
        return {"success": True, "message": "Cancelled successfully"}
    except Exception as e:
        logger.error(f"[CAL] cancel_booking error: {e}")
        return {"success": False, "message": str(e)}
