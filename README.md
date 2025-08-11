# 🛎️ Hotel FAQ & Sentiment Chatbot (Offline)

A **Streamlit-based hospitality chatbot** built with the intention to later upgrade into a fully functional **AI Agent** for hotel operations.  

This version runs entirely **offline** — no API keys or internet connection required.  
It is designed as the first step towards a modular AI Agent architecture, focusing on two core capabilities:

- **FAQ Retrieval** using a custom-created RAG (Retrieval-Augmented Generation) JSON database of hotel FAQs.
- **Sentiment Analysis** for restaurant food reviews using a locally saved machine learning model.

---

## 🎯 Intention

The main goal is to:
1. Build a **modular chatbot** that can handle hotel guest interactions without relying on external APIs.
2. Use **locally stored models and data** 
3. Establish a **scalable structure** so this chatbot can evolve into an AI Agent capable of:
   - Handling bookings
   - Managing guest feedback
   - Sending real-time offers
   - Integrating with hotel systems

---

## 🧠 Models Used

This project uses a **pre-trained sentiment analysis model** to classify restaurant food reviews as Positive, Negative, or Neutral.  
The model is stored locally as:

- `models/sentiment_model.pkl` – The trained classifier
- `models/vectorizer.pkl` – The text preprocessing vectorizer

Both files are loaded at runtime, so the chatbot can predict sentiment.

---

## 📂 RAG JSON Database

For FAQ responses, I created a **custom JSON database** stored at:

- `data/rag_data/hotel_faq.json`

This database contains question–answer pairs about hotel services, amenities, and policies.  
The chatbot uses **TF-IDF** and **cosine similarity** to retrieve the most relevant answer to guest queries.

Example entry in `hotel_faq.json`:
```json
[
  {
    "question": "What time is check-in?",
    "answer": "Check-in starts at 3:00 PM. Early check-in is subject to availability."
  }
]
