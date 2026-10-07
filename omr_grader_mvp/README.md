# OMR Grader MVP

A simple offline-first prototype that:

- generates a printable 20-question A-D answer sheet;
- scans a completed answer key first;
- validates blank, unclear, or multiple-marked key answers;
- scans student answer-sheet images or phone-camera photos;
- measures inner-bubble ink density;
- highlights uncertain questions for teacher review;
- grades against the confirmed answer key;
- downloads question results, a marked image, and session results.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Start the app:

```bash
streamlit run app.py
```

Streamlit will open the app in your browser, usually at `http://localhost:8501`.

## Run on Replit

1. Create a new Python Repl.
2. Upload all files from this folder.
3. Install packages from `requirements.txt`.
4. Use this run command:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8080
```

## Recommended test sequence

1. Open **Home & sheet** and download the answer-sheet PNG.
2. Print it several times.
3. Fill one copy as the answer key.
4. Open **Create answer key**, upload or photograph the key, review flags, and save it.
5. Fill another copy as a student sheet.
6. Open **Grade student paper**, upload it, review uncertain answers, and download the result.

## Important prototype limitation

This MVP uses a fixed sheet layout. It expects:

- 20 questions;
- four choices: A, B, C, D;
- the supplied registration squares and bubble positions;
- the entire sheet to be visible in the image.

Future versions can add customizable templates, student rosters, batch grading, local database storage, Android packaging, and teacher accounts.

## Camera document scanner

Camera captures now pass through `document_scanner.py` before OMR grading. The scanner attempts to:

- detect the full paper boundary,
- crop and perspective-correct the page,
- normalize uneven lighting and shadows,
- improve local contrast while preserving pencil/pen marks,
- then use the existing four registration squares for fine alignment.

Uploaded JPG/PNG files keep the original upload path, which is useful for images already processed by CamScanner or another scanning app.

## Bubble scoring update

The camera scanner now keeps two internal versions of each photo:

- an **aggressive white scanner preview** for a clean CamScanner-style view;
- a **detail-preserving analysis image** used by OMR so faint pencil is not erased by whitening.

For each question the OMR engine measures only the inner part of A-D, corrects each score against the paper immediately around that bubble, then compares the strongest bubble with the second strongest. A clearly dominant mark is accepted. Weak, close, or double-marked rows are flagged for review instead of being guessed.

## Two-stage perspective correction (v3)
Camera scans now use two separate geometry stages:
1. Rough document boundary crop/perspective correction.
2. Fine perspective correction using all four black registration squares.

The registration-square pass is repeated after the first warp to remove residual slant. A geometry quality score checks residual marker error and edge tilt before grading. If the final marker geometry cannot be verified, the app asks for a retake rather than trusting a bad alignment.

## v4 custom answer-sheet builder

The Home page now includes a scan-aware answer-sheet builder:
- 1-100 questions
- A-E by default (A-D optional)
- teacher-defined header/title
- Name line only
- pure white sheet background (no gray row shading)
- automatic 1/2/3/4-column layout
- four fixed black registration squares
- the selected custom layout is also used by answer-key and student scanning

Changing the sheet layout clears the existing answer key so a key made for one geometry cannot accidentally grade a different geometry.
