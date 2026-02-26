import os
import logging
from supabase import create_client, Client

logger = logging.getLogger("db")

def get_supabase() -> Client | None:
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        return None

def save_call_log(
    phone: str,
    duration: int,
    transcript: str,
    summary: str = "",
    recording_url: str = "",
    caller_name: str = "",
    sentiment: str = "unknown",
    estimated_cost_usd: float | None = None,
    call_date: str | None = None,
    call_hour: int | None = None,
    call_day_of_week: str | None = None,
    was_booked: bool = False,
    interrupt_count: int = 0,
) -> dict:
    """Saves a call log to the 'call_logs' table in Supabase."""
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        logger.info(f"Supabase not configured. Local Log -> Phone: {phone}, Duration: {duration}s")
        return {"success": False, "message": "Supabase not configured"}

    supabase = get_supabase()
    if not supabase:
        return {"success": False, "message": "Supabase client failed"}

    try:
        data: dict = {
            "phone_number":      phone,
            "duration_seconds":  duration,
            "transcript":        transcript,
            "summary":           summary,
            "sentiment":         sentiment,
            "was_booked":        was_booked,
            "interrupt_count":   interrupt_count,
        }
        if recording_url:           data["recording_url"]         = recording_url
        if caller_name:             data["caller_name"]            = caller_name
        if estimated_cost_usd is not None: data["estimated_cost_usd"] = estimated_cost_usd
        if call_date:               data["call_date"]              = call_date
        if call_hour is not None:   data["call_hour"]              = call_hour
        if call_day_of_week:        data["call_day_of_week"]       = call_day_of_week

        res = supabase.table("call_logs").insert(data).execute()
        logger.info(f"Saved call log to Supabase for {phone}")
        return {"success": True, "data": res.data}
    except Exception as e:
        logger.error(f"Failed to save call log: {e}")
        return {"success": False, "message": str(e)}


def fetch_call_logs(limit: int = 50) -> list:
    """
    Fetches the latest call logs from Supabase for the UI dashboard.
    """
    supabase = get_supabase()
    if not supabase:
        return []

    try:
        res = supabase.table("call_logs").select("*").order("created_at", desc=True).limit(limit).execute()
        return res.data
    except Exception as e:
        logger.error(f"Failed to fetch call logs: {e}")
        return []


def fetch_bookings() -> list:
    """
    Fetches confirmed bookings (calls where summary contains 'Confirmed') for the calendar.
    """
    supabase = get_supabase()
    if not supabase:
        return []
    try:
        res = (
            supabase.table("call_logs")
            .select("id, phone_number, summary, created_at")
            .ilike("summary", "%Confirmed%")
            .order("created_at", desc=True)
            .limit(200)
            .execute()
        )
        return res.data
    except Exception as e:
        logger.error(f"Failed to fetch bookings: {e}")
        return []


def fetch_stats() -> dict:
    """
    Returns aggregate stats for the dashboard: total calls, bookings, avg duration, booking rate.
    """
    supabase = get_supabase()
    if not supabase:
        return {"total_calls": 0, "total_bookings": 0, "avg_duration": 0, "booking_rate": 0}
    try:
        all_res = supabase.table("call_logs").select("duration_seconds, summary").execute()
        rows = all_res.data or []
        total = len(rows)
        bookings = sum(1 for r in rows if r.get("summary") and "Confirmed" in r.get("summary", ""))
        durations = [r["duration_seconds"] for r in rows if r.get("duration_seconds")]
        avg_dur = round(sum(durations) / len(durations)) if durations else 0
        rate = round((bookings / total) * 100) if total else 0
        return {"total_calls": total, "total_bookings": bookings, "avg_duration": avg_dur, "booking_rate": rate}
    except Exception as e:
        logger.error(f"Failed to fetch stats: {e}")
        return {"total_calls": 0, "total_bookings": 0, "avg_duration": 0, "booking_rate": 0}
