-- Swing history
create table if not exists swings (
  id uuid primary key default gen_random_uuid(),
  created_at timestamp default now(),
  user_ip text,
  score integer,
  grade text,
  handedness text,
  in_envelope_pct float,
  deviations jsonb,
  claude_feedback jsonb,
  raw_angles jsonb
);

-- Coaching response cache
create table if not exists coaching_cache (
  id uuid primary key default gen_random_uuid(),
  signature text unique,
  response jsonb,
  hit_count integer default 0,
  created_at timestamp default now()
);

-- Daily usage tracking
create table if not exists daily_usage (
  user_ip text,
  usage_date date,
  call_count integer default 0,
  primary key (user_ip, usage_date)
);

-- Atomic increment function for rate limiting
create or replace function increment_usage(p_ip text, p_date date)
returns integer as $$
declare
  new_count integer;
begin
  insert into daily_usage (user_ip, usage_date, call_count)
  values (p_ip, p_date, 1)
  on conflict (user_ip, usage_date)
  do update set call_count = daily_usage.call_count + 1
  returning call_count into new_count;
  return new_count;
end;
$$ language plpgsql;
