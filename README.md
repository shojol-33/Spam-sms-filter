# 📱 Spam SMS Filter

A machine learning project that classifies SMS messages as **Ham (legitimate)** or **Spam** using Natural Language Processing (NLP) techniques. Two classifiers — Naive Bayes and Logistic Regression — are trained and evaluated side by side.

---

## 📌 Project Overview

This notebook demonstrates a complete end-to-end spam detection pipeline:

1. **Data Loading** — Streams the SMS Spam Collection dataset directly from GitHub
2. **Preprocessing** — Maps labels to binary values (`ham = 0`, `spam = 1`)
3. **Feature Extraction** — Converts raw text to TF-IDF numerical vectors
4. **Model Training** — Trains Multinomial Naive Bayes and Logistic Regression classifiers
5. **Evaluation** — Compares accuracy, precision, recall, and F1-score for both models
6. **Prediction** — Classifies new custom messages in real time

---

## 📂 Project Structure

```
Spam_sms_filter.ipynb   # Main Jupyter Notebook
README.md               # Project documentation
requirements.txt        # Python dependencies
```

---

## 🗂️ Dataset

- **Source:** [SMS Spam Collection Dataset](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv)
- **Format:** Tab-separated (`.tsv`) with two columns: `label` and `text`
- **Classes:** `ham` (legitimate) and `spam`
- **Split:** 80% training / 20% testing (stratified)

---

## ⚙️ How It Works

### Feature Extraction
Text messages are converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)** with English stop words removed.

### Models

| Model | Notes |
|---|---|
| **Multinomial Naive Bayes** | Fast, probabilistic; works well on text data |
| **Logistic Regression** | Linear classifier with `liblinear` solver |

### Example Predictions

```
Message: "Hey, are we still meeting up for lunch at 1 PM today?"
 -> Naive Bayes: 🟢 HAM (Legit)
 -> Logistic Regression: 🟢 HAM (Legit)

Message: "URGENT! Your mobile number has won a £2,000 cash prize! Call 09051234567 now."
 -> Naive Bayes: 🚨 SPAM
 -> Logistic Regression: 🚨 SPAM
```

---

## 🚀 Getting Started

### 1. Clone or download the notebook

```bash
git clone <your-repo-url>
cd spam-sms-filter
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook Spam_sms_filter.ipynb
```

Run all cells from top to bottom. The dataset is fetched automatically — no manual download needed.

---

## 📊 Output

After running the notebook, you will see:

- Training and test set sizes
- Vocabulary size from TF-IDF
- Full classification report for both models (accuracy, precision, recall, F1)
- Live predictions on sample messages

---

## 🛠️ Requirements

See `requirements.txt` for the full list of dependencies.

---

## 📄 License

This project is for educational purposes. Dataset credits go to the original SMS Spam Collection authors.
