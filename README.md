# 📧 Email Spam Detector

A Machine Learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and Scikit-learn.

---

## Features

- Data preprocessing
- Text cleaning using Regular Expressions
- Feature extraction using CountVectorizer
- Machine Learning with Multinomial Naive Bayes
- Model evaluation using Accuracy Score and Classification Report
- Predicts whether new messages are Spam or Ham

---

## Technologies

- Python
- Pandas
- Scikit-learn
- Regular Expressions (re)

---

## Dataset

The project uses the SMS Spam Collection Dataset.

Columns:

- **v1** → Label (spam/ham)
- **v2** → Message

---

## Installation

```bash
git clone https://github.com/Kinibuhle/email-spam-detector.git

cd email-spam-detector

pip install -r requirements.txt
```

---

## Run

```bash
python spam_detector.py
```

---

## Example Output

```
Accuracy: 98.3%

Congratulations! You won R10000!
Prediction: spam

Hi Sam, are we meeting today?
Prediction: ham
```

---

## Skills Demonstrated

- Machine Learning
- Natural Language Processing (NLP)
- Data Cleaning
- Text Classification
- Feature Engineering
- Python Programming

---

## Future Improvements

- Save trained model with Joblib
- Build a Streamlit web application
- Deploy online
- Add support for email datasets
