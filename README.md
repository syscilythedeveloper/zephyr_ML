# Zephyr Innovations Chat & Analytics

This project provides a semantic search and chat engine for analyzing customer comments about Zephyr Innovations products using **FAISS** and **sentence-transformers**.

---

## 🚀 Features

- Converts text to vector embeddings using `sentence-transformers`
- Finds semantically similar comments with `FAISS`
- Filters results by:
  - Product
  - Sentiment
  - Commenter
  - Date
- Generates prompts for LLMs (via **Ollama API** integration)
- Easy-to-use Python classes for quick interaction

---

## ⚙️ Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd zephyr
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💬 Usage Example

Run sentiment analysis and query the engine:

```python
from chat_engine import ZephyrChatEngine

engine = ZephyrChatEngine(data_path="data/sentiment_results.json")

response = engine.chat("What do customers think about Rocket Skates?", k=5)
print(response["answer"])
```

With optional filters:

```python
filters = {
    "product": ["Rocket Skates"],
    "sentiment": ["POSITIVE"],
    "commenter": ["Wile E. Coyote"],
    "date_range": ("2025-01-01", "2025-07-01")
}

response = engine.chat("Any positive feedback?", k=5, filters=filters)
print(response["answer"])
```

---

## 📌 Notes

- Ensure `venv/` is excluded from version control (see `.gitignore`)
- Requires **Python 3.8+**
- **Ollama API** must be running locally for LLM prompt generation

---

## 📄 License

This project is licensed under the **MIT License**.
