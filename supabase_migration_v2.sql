-- ══════════════════════════════════════════════════════════════════════════════
-- SUPABASE MIGRATION — 40-Improvement Plan (run in SQL Editor)
-- ══════════════════════════════════════════════════════════════════════════════

-- 1. New analytics columns on call_logs (#4, #14, #19, #30, #34)
ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS sentiment       TEXT    DEFAULT NULL;
ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS estimated_cost_usd NUMERIC(10,5) DEFAULT NULL;
ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS call_date       DATE    DEFAULT NULL;
ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS call_hour       INTEGER DEFAULT NULL;
ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS call_day_of_week TEXT   DEFAULT NULL;
ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS was_booked      BOOLEAN DEFAULT FALSE;
ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS interrupt_count INTEGER DEFAULT 0;
ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS audio_codec     TEXT    DEFAULT NULL;

-- 2. Real-time transcript table (#33)
CREATE TABLE IF NOT EXISTS call_transcripts (
    id           UUID    DEFAULT gen_random_uuid() PRIMARY KEY,
    call_room_id TEXT    NOT NULL,
    phone        TEXT,
    role         TEXT    CHECK (role IN ('user', 'assistant')),
    content      TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_call_transcripts_room ON call_transcripts (call_room_id);
CREATE INDEX IF NOT EXISTS idx_call_transcripts_phone ON call_transcripts (phone);

-- Enable RLS on call_transcripts
ALTER TABLE call_transcripts ENABLE ROW LEVEL SECURITY;
CREATE POLICY IF NOT EXISTS "Allow anon insert transcripts" ON call_transcripts
    FOR INSERT TO anon WITH CHECK (true);
CREATE POLICY IF NOT EXISTS "Allow anon select transcripts" ON call_transcripts
    FOR SELECT TO anon USING (true);

-- 3. Active calls table for real-time monitoring (#38)
CREATE TABLE IF NOT EXISTS active_calls (
    room_id      TEXT PRIMARY KEY,
    phone        TEXT,
    caller_name  TEXT,
    status       TEXT DEFAULT 'ringing',
    started_at   TIMESTAMPTZ DEFAULT NOW(),
    last_updated TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE active_calls ENABLE ROW LEVEL SECURITY;
CREATE POLICY IF NOT EXISTS "Allow anon upsert active_calls" ON active_calls
    FOR ALL TO anon USING (true) WITH CHECK (true);

-- ══════════════════════════════════════════════════════════════════════════════
-- DONE. All new columns and tables are safe to run multiple times (IF NOT EXISTS).
-- ══════════════════════════════════════════════════════════════════════════════
