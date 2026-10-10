# Video outline (15 minutes, YouTube unlisted)

The judges want three things on screen: the solution, the architecture, and how it was evaluated.
They will look hardest at how the model was trained, validated and improved, so give that the most time.
Record your screen with OBS Studio or Zoom; speak in your own words rather than reading this aloud.

| Time | Show on screen | Say |
|---|---|---|
| 0:00 to 1:00 | Title slide or the demo home page | Team name, the problem: route multilingual RideEat tickets to the right team, flag urgent ones. No LLM is called; everything is a model we trained. |
| 1:00 to 3:30 | Demo site, "Route a ticket" | Click the Singlish, Sinhala, Tamil and English examples. Point at team, main issue, second issue, urgency, confidence, closest categories and the keywords picked up. Try the prompt-injection example and show it is not marked urgent. Type one ticket of your own. |
| 3:30 to 5:00 | "Route a file" | Upload `data/validation.csv`. Show the progress bar (it is a real background job), tickets per team, download CSV. Say plainly that this file was part of final training, so the match is near 100% and the honest numbers are on the next tab. |
| 5:00 to 6:00 | Architecture: `README.md` layout or a simple diagram | One container: FastAPI, the model, SQLite for jobs, one worker thread, static demo page. Key from the `API_KEY` environment variable. No internet at runtime. |
| 6:00 to 10:30 | `docs/REPORT.md` sections 2 and 3, then the "How good is it" tab | The main story. (1) Data: tickets are mostly boilerplate, and validation uses different wording from train. (2) Cross-validation says 98% for everything, so we ignored it. (3) Walk the ladder E1 to E6 and say why each step was tried. Open `app/romanize.py` and `app/lexicon.py` for ten seconds each. (4) The ideas that did not work. |
| 10:30 to 12:00 | Terminal: `python -m training.train` output, then `models/datapulse-1.0.0/model_card.json` | Training is one command. Show the printed table, the tuned thresholds, the saved artifact and its hash. `model_version` in the API equals this folder name. |
| 12:00 to 13:30 | Terminal: `pytest -q`, then `python scripts/smoke_test.py <hosted URL> <key>` (hide the key) | 58 contract tests against the organisers' schemas; live smoke test passes on the hosted URL, including a 2,500-ticket job. Mention restart handling and the 429 queue limit. |
| 13:30 to 15:00 | Report section 6 | Honest limits: 88% is optimistic because the lexicon was tuned on validation errors; typos hurt; Tanglish has few examples. What you would do next: fine-tune a multilingual encoder. |

Before recording:

- Hide the API key. The key box on the demo site is masked; in the terminal, set it in a variable
  first (`$env:API_KEY="..."` in PowerShell) and clear the screen before you start recording.
- Make sure the hosted URL is awake by opening it once.
- Do a one-minute test recording to check sound.
