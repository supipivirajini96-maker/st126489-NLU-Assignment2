
# 📘 Harry Potter LSTM Language Model

A **word-level LSTM language model** trained on the *Harry Potter* book series and deployed through a **Dash web application** that generates text continuations based on user input.

---

## 🧾 Dataset (Task 1)

The dataset consists of the **seven Harry Potter novels** by **J.K. Rowling**, each stored as a separate text file. The corpus provides long, narrative-rich text suitable for sequence modeling.

* **Source:** Kaggle public dataset
  [https://www.kaggle.com/datasets/rupanshukapoor/harry-potter-books](https://www.kaggle.com/datasets/rupanshukapoor/harry-potter-books)
* **Usage:** Educational and non-commercial only
* **Author:** J.K. Rowling

---

## 🧠 Model Training (Task 2)

### Preprocessing

* Word-level tokenization using `basic_english`
* Vocabulary construction with special tokens:

  * `<unk>` for unknown words
  * `<eos>` for sentence boundaries
* Fixed-length input sequences created using a sliding window
* User prompts are lowercased during generation for vocabulary consistency

### Model Architecture

* **Embedding layer:** 1024-dimensional word embeddings
* **2 stacked LSTM layers:** each with 1024 hidden units
* **Dropout:** 0.65 between LSTM layers
* **Output layer:** fully connected layer over the vocabulary

The model is trained using next-word prediction with cross-entropy loss.

---

## 🌐 Web Application (Task 3)

A simple **Dash web application** demonstrates the trained language model.

### Features

* Text input box for user prompts
* “Generate Continuation” button
* Output area showing only the generated continuation

### Model Interaction

1. Load trained model (`best-val-lstm_lm.pt`) and vocabulary (`vocab.json`)
2. Convert user input to tokens
3. Initialize LSTM hidden state using the prompt
4. Generate words step-by-step using temperature-based sampling
5. Display only newly generated text

---

## 🖥️ Installation & Run Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the web application

```bash
python app.py
```

### 5️⃣ Open in browser

```
http://127.0.0.1:8050/
```

---

## 🗂️ Project Structure

```text
.
├── app.py                  # Dash web application
├── model.py                # LSTM language model definition
├── vocab.json              # Vocabulary mapping
├── best-val-lstm_lm.pt     # Trained model checkpoint
├── hp_books/               # Harry Potter text files
│   ├── hp1.txt
│   ├── hp2.txt
│   ├── hp3.txt
│   ├── hp4.txt
│   ├── hp5.txt
│   ├── hp6.txt
│   └── hp7.txt
├── requirements.txt
└── README.md
```

---

## 📸 Screenshots

> *(Add screenshots after running the app and commit them to a `screenshots/` folder)*

```markdown
## 📸 Screenshots

![Web Application Interface](app/Webpage%20images/Screenshot%202026-02-01%20205423.png)

```





