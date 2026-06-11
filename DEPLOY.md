# Deploying Echo to the web (Vercel + Supabase)

Echo's cloud version is a static site + one Python serverless function on
Vercel, with Supabase providing **accounts** (Supabase Auth) and **storage**
(Postgres + pgvector). Each account's Supabase user id becomes the memlayer
`user_id`, so every user gets a fully isolated memory space.

```
 browser ──(supabase-js: sign in)──▶ Supabase Auth
    │                                      │ JWT
    ▼                                      ▼
 Vercel static (public/index.html) ──▶ /api/* (Python fn, api/index.py)
                                           │ verifies JWT → user_id
                                           ├─▶ Supabase Postgres (pgvector)
                                           └─▶ Gemini API (chat/extract/embed)
```

You need: a GitHub account, a [Supabase](https://supabase.com) account
(free tier is fine), a [Vercel](https://vercel.com) account (free tier is
fine), and a Gemini API key (https://aistudio.google.com/apikey).

## 1. Set up Supabase (~5 minutes)

1. supabase.com → **New project** (any name, choose a strong DB password).
2. Open **SQL Editor → New query**, paste the entire contents of
   [`supabase/schema.sql`](supabase/schema.sql), and **Run**. This creates
   the `memories` and `audit_log` tables, enables pgvector, adds the search
   functions, and locks both tables down with RLS (only the server can
   touch them).
3. Collect three values from **Project Settings → API**:
   - **Project URL** → `SUPABASE_URL`
   - **anon public** key → `SUPABASE_ANON_KEY`
   - **service_role** key → `SUPABASE_SERVICE_ROLE_KEY` (keep secret!)
4. Optional, for instant signups while testing: **Authentication →
   Providers → Email → disable "Confirm email"**. Leave it on for real
   deployments.

## 2. Push this repo to GitHub

```powershell
git add -A
git commit -m "Echo web deployment"
gh repo create echo-journal --private --source . --push   # or use the GitHub UI
```

## 3. Deploy on Vercel (~3 minutes)

**Via the dashboard (easiest):**

1. vercel.com → **Add New → Project** → import your GitHub repo.
2. Framework Preset: **Other** (the included `vercel.json` handles routing).
3. Under **Environment Variables**, add all four:

   | Name | Value |
   |---|---|
   | `SUPABASE_URL` | from step 1.3 |
   | `SUPABASE_ANON_KEY` | from step 1.3 |
   | `SUPABASE_SERVICE_ROLE_KEY` | from step 1.3 |
   | `GEMINI_API_KEY` | from aistudio.google.com/apikey |

4. **Deploy**. First build takes a couple of minutes (Python deps).

**Or via the CLI:**

```powershell
npm i -g vercel
vercel login
vercel                                   # link + first deploy (preview)
vercel env add SUPABASE_URL              # repeat for all four vars
vercel --prod
```

## 4. Point Supabase back at your site

In Supabase: **Authentication → URL Configuration → Site URL** = your
Vercel URL (e.g. `https://echo-yourname.vercel.app`). This makes email
confirmation links land on your site.

## 5. Try it

Open the Vercel URL → **Create account** → write an entry. Then check
Supabase **Table Editor → memories**: you'll see the episodic row plus the
facts Gemini extracted, embeddings included.

## Security model

- The browser only ever holds the **anon key** (public by design) and the
  user's own JWT. RLS is enabled with **no policies**, so the anon key
  cannot read or write the tables at all.
- The **service_role key** lives only in the Vercel function's environment.
  The function verifies the JWT on every request and uses the verified
  Supabase user id as the memlayer `user_id` — client-supplied ids are
  ignored, so accounts cannot see each other's memories.
- PII redaction is forced ON for the cloud deployment, and "off the
  record" entries are never stored.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `not signed in` (401) right after login | Site URL mismatch in Supabase Auth URL configuration; or the browser blocked third-party storage — try a normal window. |
| `No Gemini API key found` in responses | `GEMINI_API_KEY` env var missing in Vercel → add it and redeploy. |
| `relation "memories" does not exist` | You skipped step 1.2 — run `supabase/schema.sql` in the SQL editor. |
| `function match_memories does not exist` | Same — the schema file creates the RPCs; run it fully. |
| First request after idle is slow (~2-4s) | Serverless cold start (Python + numpy). Subsequent requests are fast. |
| Build exceeds size limit | Make sure `.vercelignore` is committed (it excludes `.venv`, tests, dbs). |

## Costs

Free tiers comfortably cover personal use: Vercel Hobby (serverless
functions), Supabase Free (500MB Postgres, 50k monthly active auth users),
Gemini free tier (rate-limited). The only thing that scales with heavy use
is Gemini API calls — one embedding per entry/search plus one generation
per reply/extraction.

## Local development unchanged

`echo-journal` still runs the same product fully locally (SQLite, no
accounts) — the cloud and local versions share the exact same endpoint
logic (`echo_journal/logic.py`).
