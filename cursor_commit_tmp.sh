git add api/index.py
git commit --trailer "Co-authored-by: Cursor <cursoragent@cursor.com>" -m "$(cat <<'EOF'
Harden cloud JWT auth verification against Supabase user endpoint.

Replace SDK-based token checks with direct /auth/v1/user verification so authenticated sessions can reliably access protected API routes after sign-in.
EOF
)"
git push
git status --short --branch
