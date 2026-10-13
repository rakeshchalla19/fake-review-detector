# Verification record

- Real dataset downloaded from original OSF file; publisher SHA-256 matched.
- Original OSF license relationship resolved to CC BY 4.0; metadata captured.
- Training executed on 40,432 source reviews; pipeline and test metrics saved.
- `python -m pytest -q`: **26 passed in 1.83 seconds**.
- `node --check static/app.js`: passed.
- Flask test client confirmed single inference matches the fitted pipeline, CSV sample analysis returns 3 rows / 2 products, invalid inputs fail, and rate/upload limits work.
- API health returns ready with the included model; missing model returns 503.
- User content rendered using DOM `textContent`; no submitted review database or browser storage.
- Visual browser QA and mobile-device QA: not executed in this environment; responsive styles and accessible controls are implemented, but appearance is not claimed as visually verified.
- GitHub: authenticated account verified as rakeshchalla19; target repository was not found. Available connector supports file writes but does not expose repository creation.
- Render: dashboard opened and redirected to its sign-in screen. Deployment requires sign-in and a repository connection.
- Live deployment verification: **not yet run; no live website or repository URL claimed**.
