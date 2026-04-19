"""
main.py  ─  one-click setup and run
Prepares data, trains the model, then launches the interactive predictor.

Run this first:  python main.py
"""

import os
import sys

ROOT = os.path.dirname(__file__)


def step(msg: str):
    print(f"\n{'━' * 52}")
    print(f"  {msg}")
    print(f"{'━' * 52}")


def main():
    # ── Step 1: prepare data ─────────────────────────────────────────────
    step("Step 1 / 3  ─  Preparing dataset")
    sys.path.insert(0, ROOT)
    import data.prepare_data as prep
    prep.download_dataset()

    # ── Step 2: train model ───────────────────────────────────────────────
    step("Step 2 / 3  ─  Training Naive Bayes model")
    import model.train as trainer
    trainer.main()

    # ── Step 3: launch web app ────────────────────────────────────────────
    step("Step 3 / 3  ─  Launching web app")
    print("  Open your browser at:  http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop the server.\n")
    import app as web_app
    web_app.app.run(debug=False, port=5000)


if __name__ == "__main__":
    main()