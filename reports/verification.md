# Verification record

Verified on 13 October 2026.

## Training and source
- Original OSF dataset download matched the publisher SHA-256.
- Original OSF license relationship resolved to CC BY 4.0; metadata captured.
- Training executed on 40,432 source reviews; 40,313 retained; fitted pipeline and actual held-out metrics saved.
- All 29 project files were uploaded to `rakeshchalla19/fake-review-detector` through the owner's signed-in browser. Model/source blob hashes matched local files. README and this record were subsequently updated.
- The old repository's synthetic demo files remain unused; README identifies them explicitly. The project ZIP contains the active application only.

## Automated checks
- `python -m pytest -q`: **26 passed in 3.35 seconds** on the final application.
- `node --check static/app.js`: passed.
- Tests cover real pipeline/API consistency, explanation reconstruction, input/language validation, sample CSV grouping/similarity, invalid CSVs, row/size limits, rate limiting, model unavailability and response headers.

## Deployment
- Render service: `fake-review-detector`, Free instance, Singapore.
- Render deployment `dep-db6t7pad0e5s73b98930`: **Deploy succeeded / Live**, 1m45s.
- Application commit: `932d05ad46b4344794e7e1e04609af3a00cf5a5c`.
- Python 3.12.8 detected from `.python-version`; pinned runtime dependencies installed.
- Public HTTPS URL: https://fake-review-detector-fjzd.onrender.com/
- `python scripts/verify_live.py https://fake-review-detector-fjzd.onrender.com`: passed health, real single inference and sample CSV analysis.
- Live `/api/model` JSON matched the local evaluation report exactly.
- Invalid short-review submission returned HTTP 400.

## Browser end-to-end checks
- Landing page opened over HTTPS; desktop design inspected visually.
- Single-review example submitted with product name; dashboard displayed a real margin (-0.3101) and prediction.
- Model explanation expanded and showed signed term contributions and separate indicators.
- Sample CSV downloaded from the live site and uploaded through the file chooser.
- Bulk analysis displayed 3 reviews grouped into 2 products, with flagged share and product bars.
- Similarity details displayed a request-local cosine score of 0.942 for the closest pair.
- Results CSV downloaded and parsed: 3 rows, expected product groups, predictions, margins and indicator fields.
- Model & data page displayed all three actual evaluation rows and the selected model's confusion matrix.
- These authored sample reviews have no ground-truth labels and do not establish accuracy.

## Remaining verification limits
- Mobile/tablet device testing and assistive-technology auditing were not performed.
- No load test, independent real-world deception evaluation, unseen-category evaluation or calibrated probability claim.
- Free Render sleeps during inactivity and can cold-start slowly; no uptime guarantee is made.
