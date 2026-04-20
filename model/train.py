import os
import sys
import csv
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "spam.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "naive_bayes.pkl")
VEC_PATH   = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")


def load_data(path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(row["label"].strip().lower())
            texts.append(row["message"])
    return texts, labels


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import ComplementNB
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    print("Loading data ...")
    texts, labels = load_data(os.path.abspath(DATA_PATH))
    print(f"  Total: {len(texts)} | Spam: {labels.count('spam')} | Ham: {labels.count('ham')}")

    # ── Extra spam examples ───────────────────────────────────────────────────
    extra_spam = [
        # Work from home
        "Work from home and earn 5000 weekly. No experience needed. Sign up now!",
        "Earn money from home today. Limited slots available. Join now instantly.",
        "Make money online working from home. Weekly payments guaranteed. Start now!",
        "Work at home opportunity. Earn 500 daily. No experience required. Apply now!",
        "Start earning from home today. Sign up and make money instantly. Free to join!",
        "Home based job. Earn weekly. No experience needed. Limited slots. Apply today!",
        "Get paid weekly working from home. Earn 1000 per day. Sign up free now!",
        "Make extra income from home. Weekly salary. No skills needed. Join today free!",
        "Work from home earn 5000 weekly no experience needed start today limited slots",
        "Earn 5000 per week from home. Begin making money instantly. No experience needed.",
        "Work online from home. No experience required. Earn daily. Register now free!",
        "Make money from home today. Guaranteed weekly payments. Sign up now for free!",
        "Home job opportunity. Earn 3000 weekly. No experience needed. Apply today!",
        "Online work from home. Earn 500 per day. No skills required. Join now free!",
        "Earn big working from home. Weekly payments. No experience. Sign up today!",

        # Deals and discounts
        "Exclusive deal just for you! Hurry up! Get 90% off all products. Expires in 2 hours!",
        "Flash sale! 80% off everything today only. Don't miss out! Buy now!",
        "Special offer! Hurry up and grab 70% discount before it expires. Limited time!",
        "Exclusive offer just for you. 95% off today only. Click now before it expires!",
        "Don't miss out! Huge sale ending soon. Get 85% off all items. Shop now!",
        "Hurry! Limited time deal. 90% discount expires in 1 hour. Order now!",
        "Exclusive discount for you only. Buy now and save 80%. Offer expires tonight!",
        "Last chance! 75% off everything. Deal expires in 2 hours. Don't miss it!",
        "Amazing deal! 60% off all products. Buy now before stock runs out!",
        "Biggest sale of the year! Up to 90% off. Shop now before it ends!",
        "Unbelievable offer! Get 50% off your order today. Limited time only!",
        "Today only! Massive discounts on all items. Hurry before it expires!",
        "Special discount exclusively for you. Save big today. Offer ends soon!",
        "Hot deal alert! 70% off for the next 2 hours only. Click now!",
        "Grab this deal now! 80% discount on all products. Expires midnight tonight!",
        "Hurry up! Get 90% off on all products. This deal expires in 2 hours. Don't miss out!!!",
        "Exclusive deal! 95% off everything. Limited time offer. Shop now before it ends!",
        "Subject: Exclusive Deal Just for You. Get 90% off. Hurry up expires in 2 hours!",
        "Body: Special offer just for you. 80% off all items. Don't miss this deal!",

        # Prize and lottery
        "Congratulations you won a free prize. Click here to claim your reward now!",
        "You have been selected for a free gift. Claim it now before it expires!",
        "Winner! You have won 1000 dollars. Send your details to claim your prize!",
        "You are the lucky winner of our monthly draw. Claim your prize now!",
        "Congratulations! You have won a free iPhone. Click here to claim now!",
        "You have been chosen as our winner. Reply now to receive your reward!",
        "Lucky you! You won a free vacation package. Call now to claim!",
        "You are selected as the grand prize winner. Verify your details now!",
        "Claim your free gift card worth 500 dollars. Limited time offer!",
        "You have won our sweepstakes! Click here to receive your cash prize!",
        "Congratulations! Your number was selected. Claim your reward today!",
        "You are a winner! Claim your free reward before it expires tonight!",
        "You have been randomly selected to win a prize. Claim it now!",

        # Urgency and threats
        "URGENT: Your account will be suspended. Verify your details immediately!",
        "Act now! This offer expires tonight. Don't miss your chance to save big!",
        "Limited slots available! Register now before it is too late. Hurry up!",
        "WARNING: Your account has been compromised. Click here to secure it now!",
        "FINAL NOTICE: Your subscription expires today. Renew now to avoid losing access!",
        "ALERT: Unusual activity detected on your account. Verify immediately!",
        "Your account will be deleted in 24 hours. Click here to prevent this!",
        "URGENT ACTION REQUIRED: Confirm your details or lose access forever!",
        "Last warning! Your account will be suspended. Act now to avoid this!",
        "Critical alert! Your password has been compromised. Reset it immediately!",
        "IMPORTANT: Your account is at risk. Verify your information now!",
        "Act immediately! Your account has been flagged. Confirm your details now!",

        # Financial scams
        "Investment opportunity! Double your money in 30 days. Guaranteed returns!",
        "Get rich quick! Invest 100 dollars and earn 10000 in one week!",
        "Guaranteed returns of 500 percent. Invest now and watch your money grow!",
        "Make passive income effortlessly. Invest today and earn thousands weekly!",
        "Financial freedom is one click away. Invest now for guaranteed profits!",
        "Earn 5000 dollars a day with our proven investment system. Join now!",
        "Risk free investment opportunity. Guaranteed to double your money fast!",
        "Secret investment trick banks don't want you to know. Earn big now!",
        "Millionaires are made overnight with this investment. Join us today!",
        "Exclusive investment club. Earn thousands weekly. Limited membership available!",
        "Double your investment in 7 days guaranteed. Join our exclusive program now!",
        "Proven investment strategy. Turn 500 into 50000. No risk involved. Join now!",

        # Loan and credit
        "Your loan is approved! Click this link to receive your cash instantly!",
        "Get a loan of 50000 dollars instantly. No credit check required. Apply now!",
        "Bad credit? No problem! Get instant loan approval today. Apply now!",
        "You are pre approved for a personal loan. Claim your money now!",
        "Instant cash loan available. No paperwork needed. Get money today!",
        "Emergency loan approved for you. Click here to receive funds now!",
        "Get 10000 dollars in your account today. No credit check. Apply now!",
        "Personal loan approved. No credit check needed. Get cash instantly now!",
        "Quick cash loans available. Apply now and get money in 24 hours!",

        # Health and diet scams
        "Lose 20 pounds in 2 weeks! Buy our miracle weight loss pills now!",
        "Doctors hate this trick! Lose belly fat overnight with this secret method!",
        "Revolutionary diet pill melts fat instantly. Order now and lose weight fast!",
        "Burn fat while you sleep! Our miracle supplement works overnight!",
        "Lose weight fast with our proven formula. No diet or exercise needed!",
        "Secret weight loss method doctors don't want you to know. Try now!",
        "Get slim in 7 days guaranteed. Our miracle pill works instantly!",
        "Clinical strength fat burner. Lose 30 pounds in 30 days. Order now!",
        "Amazing weight loss secret revealed. Lose 20 pounds without exercise!",
        "New diet pill approved by doctors. Lose weight fast. Order now!",

        # Phishing
        "Your bank account has been locked. Click here to unlock it now!",
        "Verify your PayPal account immediately or it will be suspended!",
        "Your credit card has been charged. Click here to dispute this charge!",
        "Security alert! Someone accessed your account. Verify your identity now!",
        "Your Netflix account will be cancelled. Update your payment details now!",
        "Click here to verify your bank details to avoid account suspension!",
        "Your Amazon account has been locked. Verify your information now!",
        "Suspicious login detected. Click here to secure your account now!",
        "Your payment failed. Update your billing information immediately!",
        "We noticed unauthorized access to your account. Verify now to secure it!",
        "Your bank requires you to verify your account details immediately!",
        "Confirm your account details to avoid permanent suspension. Click now!",

        # Adult and dating
        "Hot singles in your area want to meet you tonight! Sign up now!",
        "Beautiful women are looking for men like you. Join now for free!",
        "Meet lonely housewives in your area. Sign up free today!",
        "Find your perfect match tonight. Join our dating site for free!",
        "Singles near you are waiting. Sign up now and start chatting!",
        "Meet attractive singles in your city tonight. Join free now!",

        # Crypto scams
        "Bitcoin investment opportunity! Turn 100 into 10000 in 7 days!",
        "Crypto millionaire reveals secret trading strategy. Join now for free!",
        "Invest in Bitcoin today and earn guaranteed profits daily!",
        "Exclusive crypto trading signal. 500 percent profit guaranteed. Join now!",
        "Make thousands daily trading crypto. Our bot does all the work for you!",
        "Free crypto giveaway! Send 0.1 Bitcoin and receive 1 Bitcoin back!",
        "Elon Musk endorsed this crypto investment. Double your money today!",
        "Guaranteed crypto profits. Our AI trading bot earns 1000 daily for you!",
        "Join our exclusive crypto group. Members earn 500 percent monthly returns!",

        # Subscription traps
        "Get Netflix for free forever! Click here to claim your free subscription!",
        "Free Amazon Prime membership for life. Click here to activate now!",
        "Claim your free premium subscription today. Limited time offer!",
        "Get all streaming services for free. Click here to activate now!",
        "Free lifetime subscription to all apps. Claim it before it expires!",
        "Free premium account upgrade. Click here to activate your free subscription!",

        # Job scams
        "Hiring now! Earn 500 dollars per day working online. No experience needed!",
        "Data entry job from home. Earn 1000 weekly. No skills required. Apply now!",
        "Get paid to take surveys online. Earn 500 daily from home. Sign up free!",
        "Online typing job. Earn 50 dollars per hour. No experience needed. Join now!",
        "Part time job from home. Earn 5000 monthly. Apply now. Limited positions!",
        "Work online and earn big. No experience required. Start today. Apply now!",
        "Earn money taking online surveys. No experience needed. Join free today!",
        "Remote job available. Earn 3000 weekly. No skills required. Apply now!",

        # Miracle products
        "This miracle product will change your life forever. Order now!",
        "Doctors recommend this product for instant results. Buy now!",
        "Revolutionary product that guarantees results in 24 hours. Order today!",
        "Secret product big companies don't want you to know about. Buy now!",
        "This product sold out 3 times. Get yours before it's gone. Order now!",

        # Generic spam patterns
        "Click here now! Limited time offer. Don't miss out. Act fast!",
        "Free money waiting for you. Claim it now before it expires!",
        "You have been specially selected for this exclusive offer. Act now!",
        "This is not a joke. You have won. Click here to claim your prize!",
        "Make thousands from your phone. No experience needed. Start today!",
        "Earn passive income daily. No work required. Sign up for free now!",
        "Guaranteed money making system. Join thousands of happy members now!",
        "Stop working for others. Start your own online business today. Join free!",
        "The secret to financial freedom revealed. Click here to learn more!",
        "Join millions of people earning from home. Register now for free!",
    ]

    # ── Extra ham examples ────────────────────────────────────────────────────
    extra_ham = [
        "Hey, are you free for a call tomorrow afternoon?",
        "Please find the attached report for your review.",
        "The meeting has been rescheduled to 3pm on Friday.",
        "Can you send me the project update when you get a chance?",
        "Just checking in to see how the assignment is going.",
        "Let me know if you need any help with the homework.",
        "The invoice for last month has been processed successfully.",
        "Your order has been shipped and will arrive in 3 days.",
        "Thanks for your feedback. We will implement your suggestions.",
        "Happy to help! Let me know if you have any other questions.",
        "Hi, I wanted to follow up on our conversation from yesterday.",
        "The deadline for the project has been extended to next Monday.",
        "Please confirm your attendance for the conference next week.",
        "I have reviewed your proposal and have a few suggestions.",
        "Can we reschedule our meeting to Thursday instead?",
        "Your application has been received and is under review.",
        "The team will be having lunch together on Friday. Can you join?",
        "I have attached the documents you requested. Please review them.",
        "Thank you for attending the webinar. Here is the recording link.",
        "Your payment has been received. Thank you for your purchase.",
    ]

    texts  += extra_spam
    labels += ["spam"] * len(extra_spam)
    texts  += extra_ham
    labels += ["ham"] * len(extra_ham)

    print(f"  After augmentation: {len(texts)} | Spam: {labels.count('spam')} | Ham: {labels.count('ham')}")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=15000,
            sublinear_tf=True,
            min_df=1,
        )),
        ("nb", ComplementNB(alpha=0.05)),
    ])

    print("Training ...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()