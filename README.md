# 🎵 BeatBop — AI Music Recommendation Chatbot

BeatBop is a conversational music recommendation system that blends **long-term taste** (from your playlist) with **short-term intent** (your current mood) to surface songs you'll actually want to hear right now.

---

## ✨ Features

- **Playlist Screenshot Upload** — Upload a screenshot of any Spotify, Apple Music, or YouTube Music playlist. Groq Vision extracts the tracks automatically.
- **Taste Embedding** — Your playlist is converted into a weighted audio feature vector (by popularity), representing your long-term listening preferences.
- **Intent Vector** — Mood and energy filters generate a short-term intent signal.
- **Preference Blending** — Both signals are blended contextually:
  - Playlist only → 100% taste-driven
  - Mood/energy only → 100% intent-driven
  - Both → 60% intent + 40% taste
- **FAISS Retrieval** — Approximate nearest-neighbor search over 100k+ songs for fast, accurate candidate generation.
- **Graph-based Re-ranking** — A song similarity graph penalizes redundant candidates, improving playlist diversity.
- **ReAct Agent** — An LLM explains each recommendation referencing whether it matches your taste, mood, or both.
- **Multi-chat Sessions** — Maintain separate recommendation contexts simultaneously.

---

## 🏗️ Architecture

```
Playlist Screenshot ──→ Groq Vision ──→ Song List
                                            ↓
                                  Taste Embedding (long-term)
                                            ↓
Mood + Energy Filters ──────────→ Intent Vector (short-term)
                                            ↓
                                  Preference Blending Layer
                                            ↓
                              FAISS Retrieval (100k+ songs)
                                            ↓
                            Graph-based Re-ranking (diversity)
                                            ↓
                            ReAct LLM Agent (explanations)
```

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| UI | Streamlit |
| Vision (playlist extraction) | Groq — Llama 4 Scout |
| LLM (recommendations) | Groq — Llama 3 70B |
| Vector Search | FAISS |
| Feature Scaling | scikit-learn MinMaxScaler |
| Dataset | Spotify Tracks Dataset (Kaggle, ~100k songs) |
| Language | Python |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/beatbop.git
cd beatbop
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key
Create a file called `key2.txt` in the root directory and paste your Groq API key inside it.

```
your-groq-api-key-here
```

### 4. Add the dataset
Download the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) from Kaggle and place it in the root directory as `music_dataset.csv`.

### 5. Run the app
```bash
streamlit run app.py
```

---

## 🎛️ How to Use

1. **(Optional) Upload a playlist screenshot** from the sidebar — BeatBop will extract your tracks and build a taste profile.
2. **Set mood and energy filters** to express what you want right now.
3. **Filter by genre or artist** if needed.
4. **Chat naturally** — ask for songs for the gym, a road trip, a late night study session, anything.

---

## 📁 Project Structure

```
beatbop/
├── app.py                  # Main Streamlit app
├── music_dataset.csv       # Spotify dataset (not tracked in git)
├── key2.txt                # Groq API key (not tracked in git)
├── requirements.txt
└── README.md
```

---

## 🔒 .gitignore

Make sure your `.gitignore` includes:
```
key2.txt
music_dataset.csv
__pycache__/
*.pyc
.env
```

---

## 📌 Requirements

```
streamlit
pandas
numpy
faiss-cpu
scikit-learn
groq
```

---

## 💡 Design Decisions

**Why blend taste and intent instead of using one or the other?**
Pure mood-based filtering ignores who you are as a listener. Pure taste-based retrieval ignores what you want right now. The blend handles both — and gracefully degrades when either signal is missing.

**Why graph re-ranking on top of FAISS?**
FAISS retrieves the nearest neighbors to a single query vector, which tends to return very similar songs. The similarity graph penalizes redundant candidates so the final playlist has variety while still being relevant.

**Why Groq for vision?**
Speed. Groq's inference is fast enough to extract playlist tracks from a screenshot in under 2 seconds, keeping the UX snappy.