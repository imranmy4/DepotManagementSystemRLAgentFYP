# Deployment

Serves a **finished** trained run — no training code lives or is imported here.
Everything domain-related comes from the `depot` core package:

- `depot.inference.Policy` — load a run (model + VecNormalize + config) and pick actions
- `depot.ops.LiveOrderRun` — order → crane instruction steps (reshuffles + retrieve/place)
- `depot.env.DepotEnv` — the live yard model the policy observes

Planned layout:

```
deploy/
  api/                 # FastAPI backend: wraps Policy + LiveOrderRun
  web/                 # frontend (replaces legacy/ Streamlit dashboards)
  docker-compose.yml   # postgres (container records), redis (order queue), api, web
```

Model location is read from `DEPOT_MODELS_DIR` (default `<repo>/models`), see `depot/paths.py`.
