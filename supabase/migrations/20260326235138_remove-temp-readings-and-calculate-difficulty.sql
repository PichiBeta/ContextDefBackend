-- Remove the temp_readings prototype table and its associated triggers/functions.
-- The difficulty calculation logic lives in process-reading/calculate-difficulty.ts;
-- the standalone calculate-difficulty edge function (sole consumer of this table's
-- trigger) is being deleted alongside this migration.

-- 1. Triggers
DROP TRIGGER IF EXISTS trg_temp_readings_enqueue_difficulty ON public.temp_readings;
DROP TRIGGER IF EXISTS trg_temp_readings_set_updated_at ON public.temp_readings;

-- 2. RLS policy
DROP POLICY IF EXISTS "testing readings" ON public.temp_readings;

-- 3. Trigger function
DROP FUNCTION IF EXISTS public.temp_readings_enqueue_difficulty();

-- 4. Table (cascades remaining constraints and grants)
DROP TABLE IF EXISTS public.temp_readings;
