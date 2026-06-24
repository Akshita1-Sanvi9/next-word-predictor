# ✨ Next Word Predictor

A deep learning app that predicts the next word(s) from a seed phrase — trained on a dataset of famous quotes by Einstein, J.K. Rowling, Jane Austen, and more.

Built with **LSTM (Tensorflow)** + **Streamlit**.

---

## 🚀 Demo

Type any seed phrase → the model continues it word by word, influenced by the wisdom of the greatest minds.

> *"The world as we have"* → **created it is a process of our thinking**

---

## 🧠 How It Works

1. The quotes dataset is tokenized and converted into input sequences
2. Each sequence is padded to a fixed length (`max_len`)
3. An LSTM model is trained to predict the next word given a sequence
4. At inference, the seed text is tokenized → padded → fed to the model → top predicted word is appended → repeated N times

---

## 🗂️ Project Structure

```
next_word_pred/
├── app.py                  # Streamlit UI
├── lstm_model(1).h5        # Trained LSTM model (not pushed to GitHub)
├── tokenizer.pkl           # Fitted Keras tokenizer
├── max_len.pkl             # Max sequence length used during training
├── qoute_dataset.csv       # Quotes dataset (Author + Quote)
└── README.md
```

---

## ⚙️ Installation & Setup

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/next-word-predictor.git
cd next-word-predictor
```

**2. Install dependencies**
```bash
pip install streamlit tensorflow numpy pickle5
```

**3. Add the model file**

Download `lstm_model(1).h5` and place it in the project folder (not included in repo due to file size).

**4. Run the app**
```bash
streamlit run app.py
```

---

## 📦 Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Keras / TensorFlow | LSTM model |
| Streamlit | Web UI |
| NumPy | Array operations |
| Pickle | Saving tokenizer & max_len |

---

## 📊 Dataset

**Quotes Dataset** — a collection of famous quotes with author names.  
Includes quotes from Albert Einstein, J.K. Rowling, Jane Austen, and many more.

---

## 🔮 Future Improvements

- [ ] Temperature sampling for more creative / diverse predictions
- [ ] Train on a larger corpus for better generalization
- [ ] Deploy on Streamlit Cloud
- [ ] Add author-wise filtering (predict in style of a specific author)

---

