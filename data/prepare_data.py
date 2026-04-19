"""
prepare_data.py
Downloads the SMS Spam Collection dataset from UCI ML Repository.
If download fails, generates a small built-in sample so the project
always works offline.
"""

import os
import urllib.request
import zipfile

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00228/smsspamcollection.zip"
)
DATA_DIR = os.path.dirname(__file__)
ZIP_PATH = os.path.join(DATA_DIR, "smsspamcollection.zip")
TSV_PATH = os.path.join(DATA_DIR, "SMSSpamCollection")
CSV_PATH = os.path.join(DATA_DIR, "spam.csv")

BUILTIN_SAMPLES = [
    ("ham",  "Hey, are we still meeting for lunch tomorrow?"),
    ("ham",  "I will be late, please wait for me at the office."),
    ("ham",  "Can you send me the homework notes?"),
    ("ham",  "Happy birthday! Hope you have a great day."),
    ("ham",  "I am on my way, be there in 10 minutes."),
    ("ham",  "Did you watch the match last night?"),
    ("ham",  "Please call me when you get home."),
    ("ham",  "Let us meet at the coffee shop at 5 pm."),
    ("ham",  "The package arrived safely, thank you."),
    ("ham",  "Good morning! How are you doing today?"),
    ("spam", "FREE prize! Click here to claim your reward now!!!"),
    ("spam", "Congratulations! You have won a 1000 dollar gift card. Call now."),
    ("spam", "URGENT: Your account will be suspended. Verify immediately."),
    ("spam", "Lose 20 pounds in 2 weeks! Buy our miracle pills now!"),
    ("spam", "You are selected for a free vacation. Reply YES to claim."),
    ("spam", "Investment opportunity! Double your money guaranteed. Act now!"),
    ("spam", "Hot singles in your area want to meet you tonight!"),
    ("spam", "Your loan is approved! Click this link to receive your cash."),
    ("spam", "Limited time offer: Buy 1 get 5 FREE! Order now!"),
    ("spam", "You have been chosen! Send your bank details to claim prize."),
    ("spam", "Work from home and earn 5000 weekly. No experience needed. Sign up now!"),
    ("spam", "Earn money from home today. Limited slots available. Join now instantly."),
    ("spam", "Make money online working from home. Weekly payments guaranteed. Start now!"),
    ("spam", "Work at home opportunity. Earn 500 daily. No experience required. Apply now!"),
    ("spam", "Start earning from home today. Sign up and make money instantly. Free to join!"),
    ("spam", "Home based job. Earn weekly. No experience needed. Limited slots. Apply today!"),
    ("spam", "Get paid weekly working from home. Earn 1000 per day. Sign up free now!"),
    ("spam", "Make extra income from home. Weekly salary. No skills needed. Join today free!"),
]


def download_dataset():
    """Try to download the real UCI dataset; fall back to built-in samples."""
    try:
        print("Downloading SMS Spam Collection …")
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extract("SMSSpamCollection", DATA_DIR)
        _convert_tsv_to_csv()
        print(f"Dataset saved to: {CSV_PATH}")
    except Exception as e:
        print(f"Download failed ({e}). Using built-in sample data.")
        _write_builtin_csv()


def _convert_tsv_to_csv():
    import csv
    with open(TSV_PATH, encoding="utf-8") as f_in, \
         open(CSV_PATH, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["label", "message"])
        for line in f_in:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                writer.writerow(parts)
    print(f"Converted TSV → CSV: {CSV_PATH}")


def _write_builtin_csv():
    import csv
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "message"])
        writer.writerows(BUILTIN_SAMPLES)
    print(f"Built-in sample data written to: {CSV_PATH}")


if __name__ == "__main__":
    download_dataset()
