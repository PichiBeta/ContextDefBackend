-- Create event trigger to automatically enable RLS on all new tables in the public schema.
-- The rls_auto_enable() function was already defined in the initial remote_schema migration.

CREATE EVENT TRIGGER ensure_rls
    ON ddl_command_end
    WHEN TAG IN ('CREATE TABLE', 'CREATE TABLE AS', 'SELECT INTO')
    EXECUTE FUNCTION public.rls_auto_enable();
