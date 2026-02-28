-- ══════════════════════════════════════════════════════════════════════════════
-- SUPABASE MIGRATION v2 — Run once in Supabase SQL Editor
-- Safe to re-run: uses IF NOT EXISTS everywhere it's valid (DDL),
-- and DROP + CREATE for policies (IF NOT EXISTS not supported in Postgres RLS)
-- ══════════════════════════════════════════════════════════════════════════════

-- 1. Analytics columns on call_logs
ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS sentiment           TEXT;
ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS estimated_cost_usd  NUMERIC(10,5);
ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS call_date           DATE;
ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS call_hour           INTEGER;
ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS call_day_of_week    TEXT;
ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS was_booked          BOOLEAN DEFAULT FALSE;
ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS interrupt_count     INTEGER DEFAULT 0;
ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS audio_codec         TEXT;

-- 2. Real-time transcript table
CREATE TABLE IF NOT EXISTS call_transcripts (
    id           UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    call_room_id TEXT        NOT NULL,
    phone        TEXT,
    role         TEXT        CHECK (role IN ('user', 'assistant')),
    content      TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_call_transcripts_room  ON call_transcripts (call_room_id);
CREATE INDEX IF NOT EXISTS idx_call_transcripts_phone ON call_transcripts (phone);

ALTER TABLE call_transcripts ENABLE ROW LEVEL SECURITY;

-- Policies: DROP first (no-op if not exists), then CREATE
DROP POLICY IF EXISTS "Allow anon insert transcripts" ON call_transcripts;
CREATE POLICY "Allow anon insert transcripts"
    ON call_transcripts FOR INSERT TO anon WITH CHECK (true);

DROP POLICY IF EXISTS "Allow anon select transcripts" ON call_transcripts;
CREATE POLICY "Allow anon select transcripts"
    ON call_transcripts FOR SELECT TO anon USING (true);

-- 3. Active calls table
CREATE TABLE IF NOT EXISTS active_calls (
    room_id      TEXT        PRIMARY KEY,
    phone        TEXT,
    caller_name  TEXT,
    status       TEXT        DEFAULT 'ringing',
    started_at   TIMESTAMPTZ DEFAULT NOW(),
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE active_calls ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow anon all active_calls" ON active_calls;
CREATE POLICY "Allow anon all active_calls"
    ON active_calls FOR ALL TO anon USING (true) WITH CHECK (true);

-- ══════════════════════════════════════════════════════════════════════════════
-- DONE. You can re-run this script safely at any time.
-- ══════════════════════════════════════════════════════════════════════════════
