# Deployment

Serves one **finished** model — no training code lives or is imported here. The model is the
bundle in `deploy/model/` (put there by `training/promote.py`; path overridable with
`DEPOT_DEPLOY_MODEL_DIR`). From the `depot` core package it uses:

- `depot.inference.Policy` — load the model (+ VecNormalize + config) and pick actions
- `depot.env.DepotEnv` — the yard model the policy observes; `plan_relocations` for reshuffles

```
deploy/
  model/               # the deployed model bundle (gitignored)
  api/
    alembic/           # schema migrations
    app/
      settings.py      # .env settings
      iso6346.py       # container number validation
      adapter.py       # DB -> env -> policy -> planned crane moves
      db/              # models, session, seed (yard layout), operations (order flow)
  docker-compose.yml   # postgres; planned: redis, api, web
```
Planned: Redis queue + worker, FastAPI endpoints, `web/` frontend.
