# Working with Dean on CLIPPER

## Merging pull requests

Default: **create the PR and stop there.** Dean merges it himself on GitHub.
Do not merge PRs automatically unless he explicitly says so for that specific
PR. This applies across sessions unless he states otherwise in the moment.

## Watching PRs

When asked to watch/babysit a PR (or by default after opening one), subscribe
to PR activity and drive it to green per the standing PR rules — but never
merge it yourself; merging stays Dean's call per the rule above.

## Repo layout quirk

- `main` (and PR branches based on it) uses **flat** paths: `clipper/*.py`, `webapp/main.py`.
- The active Claude Code session branch (`claude/clipper-github-setup-pyyn2i`)
  uses **nested** paths: `clipper/clipper/*.py`, `clipper/webapp/main.py`.
- Workflow for shipping a change: commit on the session branch (nested paths) →
  push → branch a new PR branch off `origin/main` → `git cherry-pick` the commit.
  A brand-new file will hit "CONFLICT (file location)" — resolve with
  `git show <hash>:clipper/clipper/<file>.py > clipper/<file>.py && git add clipper/<file>.py && git cherry-pick --continue --no-edit`.
  Modified files resolve automatically via content-based rename detection.
- After checking out back to the session branch, `webapp/__pycache__` is
  sometimes left behind as a stray artifact — safe to `rm -rf webapp` to
  clean it up (that directory isn't tracked on the session branch).

## Railway deploys

- Project `12720d82-23db-498d-b4f6-2a69cf5fff42`, service
  `978f7dca-a5bf-4663-8446-ed3a6625841f`, environment
  `a30ad832-9974-4645-bed0-37002a3af844` (production).
- Merges to `main` auto-deploy. After a merge, check `get-logs` /
  `environment-status` to confirm the deploy went out clean before telling
  Dean it's live.

## Other standing preferences

- No paid signups/subscriptions for libraries or tools — check for a free
  option first (e.g. plain CSS charts instead of a charting library).
- Dean posts ~2 clips/day, chosen manually by eye — factor this into any
  quota/cost estimates (YouTube API, Claude API, Railway usage, etc.).
