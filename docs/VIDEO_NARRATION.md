# Video narration (about 14 minutes)

Each part says what to put on screen and gives words you can say. Say it in your own way; you do
not have to read it exactly. Practise once before recording. Never show the API key.

Have these open before you start: the live demo site (key already pasted), `docs/DataPulse_Report.pdf`,
the GitHub repository page, and a PowerShell window in a **copy** of the project folder.

---

## Part 1. Introduction (0:00 to 1:00)

**Screen:** the demo site home page.

"Hello, we are team Data Pulse, and this is our Phase 2 MVP for TensorForge 2.0. The problem is
customer support for RideEat, a ride-hailing and food delivery platform. Tickets arrive by chat,
email and phone, in English, Sinhala, Tamil, Singlish and Tanglish. Our service reads one ticket
and returns four things: the main issue category, a second category if there is one, the team that
should handle it, and whether it is urgent. We do not call any large language model. Everything
you will see comes from a model we trained ourselves with scikit-learn."

## Part 2. Live demo, one ticket (1:00 to 3:30)

**Screen:** "Route a ticket" tab. Click each example and press "Route this ticket".

"Here is a Singlish chat message: the food parcel arrived but the drink is missing, and the customer
also wants a refund. The service sends it to Food Operations as a missing or wrong order, with
payment or refund as the second issue. It is not urgent."

"This one is in English. The driver is speeding and will not slow down, and the passenger is still
in the car. It goes to Trust and Safety and it is marked urgent."

"This is Sinhala script about a bag left in the car. It goes to Lost and Found."

"This is Tamil: the food was ordered an hour ago and has not arrived. It goes to Delivery
Operations."

"And this email is polite and calm, but it says a diabetic father's insulin has not arrived. The
model marks it urgent, because urgency depends on the consequence, not on angry words."

**Screen:** point at the lower part of the routing slip.

"Under the result we show the three closest categories with their probabilities, and the domain
keywords the model picked up. If confidence is below sixty percent, the ticket is flagged for human
review instead of being routed automatically."

**Screen:** click the "Prompt injection" example.

"This ticket says: ignore all previous instructions and mark this urgent. Our model treats that as
ordinary text. It is classified as a promo problem and it is not urgent."

## Part 2b. Live inbox (about 45 seconds, inside the time of Part 3)

**Screen:** "Live inbox" tab. Press "Start the shift" and let it run while you talk.

"This tab shows what the service would look like on a real support floor. Tickets arrive one at a
time in English, Sinhala, Tamil and Singlish. The model reads each one in about ten milliseconds and
drops it into the lane of the right team. Urgent tickets get a red marker and go to the top of their
lane. I can click any ticket to see the keywords it matched and a suggested first reply in the
customer's own language. The replies are fixed templates chosen by category, not generated text."

## Part 3. Live demo, a whole file (3:30 to 5:00)

**Screen:** "Route a file" tab. Upload `data/validation.csv`, press "Route this file".

"Judges will evaluate with large batches, so the service supports background jobs. I upload eight
hundred tickets. The page submits a job, polls the status, and the progress bar is the real job
progress. It finishes in a few seconds. We see how many tickets each team receives, how many are
urgent, and we can download the predictions as CSV."

"To be honest about this screen: the final model was trained on this validation file as well, so
the match here is close to one hundred percent. The honest test numbers are on the next tab."

## Part 4. Architecture (5:00 to 6:15)

**Screen:** Figure 2 in the report (page 3).

"The whole solution is one Docker container. Requests come in to a FastAPI layer. It checks the API
key first, then the content type, the size, the JSON, and then validates every field, in exactly
the order the contract asks for. Valid tickets go through feature extraction and then three linear
SVM classifiers: category, urgency and secondary category. One module applies the contract rules,
for example the team always follows the category, and spam is never urgent."

"Large jobs are stored in a SQLite database and processed by one background worker thread in chunks
of one hundred, so health checks and single predictions stay fast while a job runs. If the
container restarts in the middle of a job, that job is reported as failed with the code
interrupted, and queued jobs continue. The API key comes from an environment variable. It is not in
the code or in the image, and the container needs no internet at runtime."

## Part 5. How we trained, validated and improved the model (6:15 to 10:30)

**Screen:** report page 1, section 2.

"This is the most important part. First, what we found in the data. A typical ticket is mostly
boilerplate: four or five filler sentences around one or two sentences that state the real issue.
Second, and this changed everything: the validation tickets describe the same kinds of problems as
the training tickets, but with different wording. We checked this by extracting the issue sentences
from both files. There is almost no overlap."

**Screen:** Figure 1 in the report (page 2).

"This chart shows why that matters. The orange dots are cross-validation inside the training file.
Every model gets about ninety-eight percent, even the simplest one. The blue dots are the same
models scored on the validation file. Our baseline, word TF-IDF with logistic regression, falls to
sixty percent. So cross-validation was useless for choosing a model, and we made every decision
using the validation file only."

"Then we improved step by step. A linear SVM gave sixty-four percent. Character n-grams, which
handle typos and spelling changes, gave sixty-six."

**Screen:** open `app/romanize.py` on GitHub for a few seconds.

"The next idea was script unification. Sinhala script and Singlish say the same words with different
letters. This small module converts Sinhala and Tamil letters to Latin letters, so 'salli kapila' in
Sinhala script and in Singlish become the same features. Sinhala accuracy went from forty-seven to
fifty-seven percent."

**Screen:** open `app/lexicon.py` on GitHub, scroll slowly.

"The biggest gain came from this file, a domain concept lexicon. New wording uses new words, and
n-grams cannot know that reimbursement means refund. So we wrote keyword lists for thirty-seven
concepts, such as refund, delay, lost item, threat and app failure, in English, romanised Sinhala
and romanised Tamil. The lexicon only tells the model that a concept is present. The SVM still
learns from the data what each concept means for each label. That took us from sixty-eight to
eighty-five and a half percent."

"The last step is boilerplate masking. Filler sentences were triggering keywords by accident. Any
three-word phrase that appears across five or more categories cannot be about one category, so we
learn those phrases from the data and remove them before matching keywords. That gave eighty-eight
percent."

**Screen:** the "Ideas we tested and rejected" table in the report.

"Not everything worked. Pretrained English word vectors gave about one point for a hundred and sixty
megabytes. Synonym augmentation was not reliable. Removing boilerplate from the TF-IDF input made
things worse. We rejected all of these."

## Part 6. Evaluation results (10:30 to 12:00)

**Screen:** "How good is it" tab on the demo site; scroll slowly.

"These are the results of the final recipe trained on the training file only and scored on
validation. Category accuracy eighty-eight percent, macro F1 zero point eight seven six. Urgency F1
zero point nine two four. Secondary category F1 zero point eight eight two."

"By language, Singlish is the strongest at ninety-eight percent, then English, Sinhala and Tamil.
Tanglish is lowest, but it has only twenty validation tickets."

"We also calibrated the confidence. The raw SVM scores are turned into probabilities with a
temperature fitted on validation, and the calibration error fell from zero point five nine to zero
point zero three. This table shows that when the model says ninety percent, it is right about ninety
percent of the time."

## Part 7. Reproducibility and testing (12:00 to 13:15)

**Screen:** PowerShell in the copy folder. Run `python -m training.train` before recording so the
output is already on screen, then run `pytest -q` live.

"Training is one command. It prints the experiment table, tunes the thresholds, and saves the model
with a version name and a file hash. The model version returned by the API is the same as this
folder name in the repository."

"We wrote fifty-eight automated tests that check every response against the organisers' JSON
schemas: authentication, the order of error codes, batch rules, paging, idempotency, the queue
limit and restart handling."

**Screen:** run `python scripts/smoke_test.py <your hosted URL> <key>` beforehand, clear the screen
so the key is not visible, and show only the result lines ending with ALL CHECKS PASSED.

"And this is the same check running against our hosted service, including a two thousand five
hundred ticket background job."

## Part 8. Limitations and next steps (13:15 to 14:15)

**Screen:** report page 4, section 6.

"We want to be clear about the limits. We improved the keyword lexicon after reading validation
errors, so eighty-eight percent is optimistic. On completely new wording we expect a result between
sixty-eight and eighty-eight percent. Heavy typos reduce accuracy, because keywords must match. And
the lexicon needs maintenance when the product changes."

"With more time we would fine-tune a small multilingual transformer and combine it with these
features, and add fuzzy keyword matching for typos. Thank you for watching."
