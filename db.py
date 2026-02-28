import os
import time
import logging
from supabase import create_client, Client

logger = logging.getLogger("db")

# ─── Columns added by supabase_migration_v2.sql ───────────────────────────────
# If the migration hasn't been run yet, these columns won't exist.
# We detect PGRST204 (schema cache miss) and retry with just base columns.
_ANALYTICS_COLUMNS = {
    "sentiment", "was_booked", "interrupt_count",
    "estimated_cost_usd", "call_date", "call_hour", "call_day_of_week",
}
_BASE_COLUMNS = {"phone_number", "duration_seconds", "transcript", "summary",
                 "recording_url", "caller_name"}

# ─── Retry helper ─────────────────────────────────────────────────────────────
_MAX_RETRIES = 3
_RETRY_DELAYS = [1.0, 2.0, 4.0]   # seconds — covers transient SSL 525 errors


def _is_retryable(err_str: str) -> bool:
    """True if the error is a transient network or SSL failure worth retrying."""
    transient = ("525", "ssl", "timeout", "connection", "network", "502", "503", "504")
    el = err_str.lower()
    return any(k in el for k in transient)


def _is_schema_error(err_str: str) -> bool:
    """True if Supabase returned PGRST204 — column not found in schema cache."""
    return "PGRST204" in err_str or "schema cache" in err_str.lower()


# ─── Client ───────────────────────────────────────────────────────────────────

def get_supabase() -> Client | None:
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        logger.error(f"Failed to init Supabase client: {e}")
        return None


# ─── save_call_log ────────────────────────────────────────────────────────────

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
    """
    Insert a call log into Supabase.

    Strategy:
    1. Try with all columns (including analytics columns from migration_v2).
    2. If PGRST204 (column not in schema cache — migration not yet run),
       retry with only the base columns so the call is never silently lost.
    3. Retry up to 3× on transient SSL/network errors with exponential backoff.
    """
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        logger.info(f"Supabase not configured. Local log → {phone} {duration}s")
        return {"success": False, "message": "Supabase not configured"}

    supabase = get_supabase()
    if not supabase:
        return {"success": False, "message": "Supabase client failed"}

    # Build full payload
    full_data: dict = {
        "phone_number":    phone,
        "duration_seconds": duration,
        "transcript":      transcript,
        "summary":         summary,
        "sentiment":       sentiment,
        "was_booked":      was_booked,
        "interrupt_count": interrupt_count,
    }
    if recording_url:               full_data["recording_url"]      = recording_url
    if caller_name:                 full_data["caller_name"]         = caller_name
    if estimated_cost_usd is not None: full_data["estimated_cost_usd"] = estimated_cost_usd
    if call_date:                   full_data["call_date"]           = call_date
    if call_hour is not None:       full_data["call_hour"]           = call_hour
    if call_day_of_week:            full_data["call_day_of_week"]    = call_day_of_week

    # Base-only payload (fallback if migration not run)
    base_data: dict = {k: v for k, v in full_data.items() if k not in _ANALYTICS_COLUMNS}

    def _try_insert(data: dict, label: str) -> dict:
        for attempt in range(_MAX_RETRIES):
            try:
                res = supabase.table("call_logs").insert(data).execute()
                logger.info(f"Saved call log for {phone} ({label})")
                return {"success": True, "data": res.data}
            except Exception as e:
                err = str(e)
                if _is_schema_error(err):
                    # Column missing — propagate so caller can retry with base
                    raise RuntimeError("SCHEMA_ERROR:" + err)
                if _is_retryable(err) and attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_DELAYS[attempt]
                    logger.warning(f"Transient error (attempt {attempt+1}), retrying in {delay}s: {err[:80]}")
                    time.sleep(delay)
                    continue
                logger.error(f"Failed to save call log ({label}): {e}")
                return {"success": False, "message": err}
        return {"success": False, "message": "Max retries exceeded"}

    # Attempt 1: full payload
    try:
        return _try_insert(full_data, "full")
    except RuntimeError as e:
        err = str(e)
        if "SCHEMA_ERROR" in err:
            # Migration not run yet — fall back to base columns only
            logger.warning(
                "Analytics columns missing (run supabase_migration_v2.sql). "
                "Falling back to base columns for this call log."
            )
            return _try_insert(base_data, "base-fallback")
        raise


# ─── fetch_call_logs ──────────────────────────────────────────────────────────

def fetch_call_logs(limit: int = 50) -> list:
    supabase = get_supabase()
    if not supabase:
        return []
    for attempt in range(_MAX_RETRIES):
        try:
            res = (
                supabase.table("call_logs")
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return res.data
        except Exception as e:
            if _is_retryable(str(e)) and attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAYS[attempt])
                continue
            logger.error(f"Failed to fetch call logs: {e}")
            return []
    return []


# ─── fetch_bookings ───────────────────────────────────────────────────────────

def fetch_bookings() -> list:
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


# ─── fetch_stats ──────────────────────────────────────────────────────────────

def fetch_stats() -> dict:
    _empty = {"total_calls": 0, "total_bookings": 0, "avg_duration": 0, "booking_rate": 0}
    supabase = get_supabase()
    if not supabase:
        return _empty
    try:
        rows = (supabase.table("call_logs").select("duration_seconds, summary").execute()).data or []
        total = len(rows)
        bookings = sum(1 for r in rows if "Confirmed" in r.get("summary", ""))
        durations = [r["duration_seconds"] for r in rows if r.get("duration_seconds")]
        avg_dur = round(sum(durations) / len(durations)) if durations else 0
        rate = round((bookings / total) * 100) if total else 0
        return {"total_calls": total, "total_bookings": bookings, "avg_duration": avg_dur, "booking_rate": rate}
    except Exception as e:
        logger.error(f"Failed to fetch stats: {e}")
        return _empty
