-- =============================================================================
-- Echo / memlayer schema for Supabase (Postgres + pgvector)
-- Run this once in the Supabase SQL editor (Database -> SQL Editor -> New query)
-- =============================================================================

create extension if not exists vector;

-- ---------------------------------------------------------------- memories
create table if not exists public.memories (
    id               text primary key,
    memory_type      text not null,
    content          text not null,
    user_id          text not null,
    agent_id         text,
    session_id       text,
    importance       double precision not null default 0.5,
    category         text,
    metadata         jsonb not null default '{}'::jsonb,
    source_ids       jsonb not null default '[]'::jsonb,
    created_at       double precision not null,
    updated_at       double precision not null,
    last_accessed_at double precision not null,
    access_count     integer not null default 0,
    strength         double precision not null default 1.0,
    valid_from       double precision,
    valid_until      double precision,
    superseded_by    text,
    embedding        vector(768),
    content_tsv      tsvector generated always as
                       (to_tsvector('english', content)) stored
);

create index if not exists idx_mem_user    on public.memories (user_id, memory_type);
create index if not exists idx_mem_session on public.memories (session_id);
create index if not exists idx_mem_valid   on public.memories (valid_until);
create index if not exists idx_mem_tsv     on public.memories using gin (content_tsv);
create index if not exists idx_mem_vec     on public.memories
    using hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------- audit log
create table if not exists public.audit_log (
    id        text primary key,
    ts        double precision not null,
    user_id   text,
    action    text not null,
    memory_id text,
    reasoning text,
    detail    jsonb not null default '{}'::jsonb
);

create index if not exists idx_audit_mem on public.audit_log (memory_id);
create index if not exists idx_audit_ts  on public.audit_log (ts);

-- ------------------------------------------------------------------ security
-- RLS is enabled with NO policies: the anon/authenticated keys cannot touch
-- these tables at all. Only the service-role key (used by the Vercel
-- function, server-side) can read/write; per-user scoping is enforced there.
alter table public.memories  enable row level security;
alter table public.audit_log enable row level security;

-- ------------------------------------------------------------ rpc: vector
create or replace function public.match_memories(
    query_embedding vector(768),
    match_limit     int,
    p_user_id       text             default null,
    p_agent_id      text             default null,
    p_session_id    text             default null,
    p_memory_type   text             default null,
    p_current_only  boolean          default true,
    p_as_of         double precision default null
)
returns table (
    id text, memory_type text, content text, user_id text, agent_id text,
    session_id text, importance double precision, category text,
    metadata jsonb, source_ids jsonb, created_at double precision,
    updated_at double precision, last_accessed_at double precision,
    access_count integer, strength double precision,
    valid_from double precision, valid_until double precision,
    superseded_by text, similarity double precision
)
language sql stable
as $$
    select m.id, m.memory_type, m.content, m.user_id, m.agent_id,
           m.session_id, m.importance, m.category, m.metadata, m.source_ids,
           m.created_at, m.updated_at, m.last_accessed_at, m.access_count,
           m.strength, m.valid_from, m.valid_until, m.superseded_by,
           1 - (m.embedding <=> query_embedding) as similarity
    from public.memories m
    where m.embedding is not null
      and (p_user_id     is null or m.user_id     = p_user_id)
      and (p_agent_id    is null or m.agent_id    = p_agent_id)
      and (p_session_id  is null or m.session_id  = p_session_id)
      and (p_memory_type is null or m.memory_type = p_memory_type)
      and (case
             when p_as_of is not null then
               (m.valid_from  is null or m.valid_from  <= p_as_of) and
               (m.valid_until is null or m.valid_until >  p_as_of)
             when p_current_only then m.valid_until is null
             else true
           end)
    order by m.embedding <=> query_embedding
    limit match_limit;
$$;

-- ------------------------------------------------------------ rpc: keyword
create or replace function public.search_memories_text(
    search_query   text,
    match_limit    int,
    p_user_id      text             default null,
    p_agent_id     text             default null,
    p_session_id   text             default null,
    p_memory_type  text             default null,
    p_current_only boolean          default true,
    p_as_of        double precision default null
)
returns table (
    id text, memory_type text, content text, user_id text, agent_id text,
    session_id text, importance double precision, category text,
    metadata jsonb, source_ids jsonb, created_at double precision,
    updated_at double precision, last_accessed_at double precision,
    access_count integer, strength double precision,
    valid_from double precision, valid_until double precision,
    superseded_by text
)
language sql stable
as $$
    select m.id, m.memory_type, m.content, m.user_id, m.agent_id,
           m.session_id, m.importance, m.category, m.metadata, m.source_ids,
           m.created_at, m.updated_at, m.last_accessed_at, m.access_count,
           m.strength, m.valid_from, m.valid_until, m.superseded_by
    from public.memories m
    where m.content_tsv @@ websearch_to_tsquery('english', search_query)
      and (p_user_id     is null or m.user_id     = p_user_id)
      and (p_agent_id    is null or m.agent_id    = p_agent_id)
      and (p_session_id  is null or m.session_id  = p_session_id)
      and (p_memory_type is null or m.memory_type = p_memory_type)
      and (case
             when p_as_of is not null then
               (m.valid_from  is null or m.valid_from  <= p_as_of) and
               (m.valid_until is null or m.valid_until >  p_as_of)
             when p_current_only then m.valid_until is null
             else true
           end)
    order by ts_rank(m.content_tsv,
                     websearch_to_tsquery('english', search_query)) desc
    limit match_limit;
$$;

-- ------------------------------------------------------- rpc: reinforcement
create or replace function public.touch_memories(
    p_ids    text[],
    p_factor double precision,
    p_max    double precision
)
returns void
language sql
as $$
    update public.memories
    set last_accessed_at = extract(epoch from now()),
        access_count     = access_count + 1,
        strength         = least(strength * p_factor, p_max)
    where id = any(p_ids);
$$;

-- ------------------------------------------------------------- rpc: users
create or replace function public.distinct_users()
returns table (user_id text)
language sql stable
as $$
    select distinct user_id from public.memories order by user_id;
$$;
