#importing necessary python libraries
import re
try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    # pandas might also be missing; we'll fall back to csv reader when needed
    import csv

# Load the dataset with Pandas and inspect the data.
# The provided CSV has extra empty columns and headers 'v1','v2' for label and message.
if SKLEARN_AVAILABLE:
    datafile = pd.read_csv('spam.csv', encoding='latin-1')
    # Keep only the first two columns (label and message) which are named 'v1' and 'v2'
    if set(['v1', 'v2']).issubset(datafile.columns):
        datafile = datafile[['v1', 'v2']].copy()
        datafile.columns = ['label', 'message']
    else:
        # Fallback: take first two columns by position
        datafile = datafile.iloc[:, :2].copy()
        datafile.columns = ['label', 'message']
    print("Dataset Shape:", datafile.shape)
    print(datafile.head())
else:
    # Minimal CSV load without pandas
    data_rows = []
    with open('spam.csv', 'r', encoding='latin-1') as f:
        reader = csv.reader(f)
        for row in reader:
            # skip empty rows
            if not row:
                continue
            # try to take first two columns
            label = row[0] if len(row) > 0 else ''
            message = row[1] if len(row) > 1 else ''
            data_rows.append((label, message))
    # skip header if it contains 'v1'
    if data_rows and data_rows[0][0].lower().startswith('v1'):
        data_rows = data_rows[1:]
    print(f"Loaded {len(data_rows)} rows from CSV (no pandas)")


# Basic preprocessing function
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = s.lower()
    # remove non-alphanumeric characters (keep spaces)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# Apply preprocessing
if SKLEARN_AVAILABLE:
    datafile['message'] = datafile['message'].astype(str).apply(clean_text)

    # Features and labels
    X = datafile['message']
    y = datafile['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Build a model
    model = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])

    # Training the model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluating
    accuracy = accuracy_score(y_test, predictions)

    # Printing the accuracy and classification report
    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Test messages
    samples = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
        "U dun say so early hor... U c already then say...",
    ]

    print("\nTesting messages:")
    for msg in samples:
        cleaned = clean_text(msg)
        result = model.predict([cleaned])[0]
        print(f"{msg} --> {result}")
else:
    # Fallback simple keyword-based classifier
    labels = [lbl for lbl, _ in data_rows]
    messages = [clean_text(msg) for _, msg in data_rows]

    # Build simple keyword list
    spam_keywords = set([
        'free', 'win', 'winner', 'claim', 'urgent', 'prize', 'cash', 'call', 'txt', 'text', 'congrats', 'won', 'subscription', 'reply', 'mobile'
    ])

    def predict_keyword(msg: str) -> str:
        tokens = set(msg.split())
        # if any spam keyword in tokens -> spam else ham
        return 'spam' if tokens & spam_keywords else 'ham'

    # Simple 80/20 split
    n = len(messages)
    split = int(n * 0.8)
    X_train_msgs = messages[:split]
    y_train = labels[:split]
    X_test_msgs = messages[split:]
    y_test = labels[split:]

    preds = [predict_keyword(m) for m in X_test_msgs]

    # evaluation: accuracy, precision, recall, f1 for spam
    def compute_metrics(y_true, y_pred, positive='spam'):
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == positive and yp == positive)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != positive and yp == positive)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == positive and yp != positive)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    correct = sum(1 for a, b in zip(y_test, preds) if a == b)
    acc = correct / len(y_test) if y_test else 0.0
    prec, rec, f1 = compute_metrics(y_test, preds, positive='spam')

    print(f"\nFallback classifier results (keyword-based):")
    print(f"Test set size: {len(y_test)}")
    print(f"Accuracy: {acc:.2%}")
    print(f"Spam precision: {prec:.2%}, recall: {rec:.2%}, f1: {f1:.2%}")

    # show some sample predictions
    samples = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
        "U dun say so early hor... U c already then say...",
    ]
    print("\nTesting messages:")
    for msg in samples:
        print(f"{msg} --> {predict_keyword(clean_text(msg))}")


