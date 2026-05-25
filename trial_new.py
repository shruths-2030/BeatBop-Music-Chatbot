import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
import PyPDF2
import io
import json
import faiss
from sklearn.preprocessing import MinMaxScaler

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="BeatBop", page_icon="🎵", layout="wide")
st.title("🎵 BeatBop - Music Recommendation Bot")

# ─── Init Groq Client ──────────────────────────────────────────────────────────
@st.cache_resource
def init_client():
    with open("key2.txt", "r") as f:
        return Groq(api_key=f.read().strip())

client = init_client()

@st.cache_data(ttl=300)
def get_available_models():
    try:
        models = client.models.list()
        working_models = [
            m.id for m in models.data
            if any(x in m.id for x in ['llama3', 'mixtral', 'gemma', 'llama-4'])
        ]
        return working_models[:3] if working_models else ["llama3-70b-8192"]
    except:
        return ["llama3-70b-8192"]

# ─── Load & Index Dataset ──────────────────────────────────────────────────────
FEATURE_COLS = ['danceability', 'energy', 'valence', 'tempo',
                'acousticness', 'instrumentalness', 'liveness', 'speechiness']

@st.cache_resource
def load_and_index():
    df = pd.read_csv("music_dataset.csv")
    df = df.dropna(subset=FEATURE_COLS + ['track_name', 'artists', 'track_genre'])
    df = df.drop_duplicates(subset=['track_name', 'artists']).reset_index(drop=True)

    scaler = MinMaxScaler()
    features = scaler.fit_transform(df[FEATURE_COLS].values).astype('float32')

    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)

    return df, index, scaler

df, faiss_index, scaler = load_and_index()
st.success(f"✅ {len(df)} songs indexed and ready")

# ─── PDF Playlist Parser ───────────────────────────────────────────────────────
def extract_songs_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def parse_playlist_with_llm(text):
    prompt = f"""Extract all song names and artists from this playlist text.
Return ONLY a JSON array like: [{{"song": "...", "artist": "..."}}]
No extra text, no markdown.

Playlist text:
{text[:3000]}"""

    resp = client.chat.completions.create(
        model=st.session_state.get("model", "llama3-70b-8192"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1000
    )
    try:
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except:
        return []

def get_playlist_feature_vector(songs):
    """Average feature vector of matched songs from the dataset."""
    vectors = []
    for item in songs:
        match = df[
            df['track_name'].str.lower().str.contains(item['song'].lower(), na=False) |
            df['artists'].str.lower().str.contains(item['artist'].lower(), na=False)
        ]
        if not match.empty:
            vectors.append(match[FEATURE_COLS].mean().values)
    if vectors:
        return np.mean(vectors, axis=0).astype('float32')
    return None

# ─── Filter → Feature Vector Mapping ──────────────────────────────────────────
MOOD_VECTORS = {
    "Happy":     dict(valence=0.8, energy=0.7, danceability=0.7),
    "Sad":       dict(valence=0.2, energy=0.3, acousticness=0.7),
    "Party":     dict(danceability=0.9, energy=0.9, valence=0.7),
    "Chill":     dict(energy=0.3, acousticness=0.6, tempo=0.3),
    "Angry":     dict(energy=0.9, valence=0.2, loudness=0.9),
    "Romantic":  dict(valence=0.6, acousticness=0.6, energy=0.4),
    "Uplifting": dict(valence=0.85, energy=0.75, danceability=0.65),
    "Focus":     dict(instrumentalness=0.7, energy=0.4, speechiness=0.1),
}

ENERGY_VECTORS = {
    "High Beat":   dict(tempo=0.9, energy=0.9, danceability=0.8),
    "Medium":      dict(tempo=0.5, energy=0.5),
    "Low / Calm":  dict(tempo=0.2, energy=0.2, acousticness=0.7),
}

def build_query_vector(mood=None, energy=None, genre=None, playlist_vec=None):
    base = {col: 0.5 for col in FEATURE_COLS}

    if mood and mood in MOOD_VECTORS:
        base.update(MOOD_VECTORS[mood])
    if energy and energy in ENERGY_VECTORS:
        base.update(ENERGY_VECTORS[energy])

    vec = np.array([base[c] for c in FEATURE_COLS], dtype='float32')

    if playlist_vec is not None:
        vec = 0.5 * vec + 0.5 * playlist_vec  # blend user taste with filters

    return vec.reshape(1, -1)

# ─── RAG Retrieval ─────────────────────────────────────────────────────────────
def retrieve_songs(query_vec, genre=None, artist_filter=None, k=50):
    scaled_vec = scaler.transform(query_vec).astype('float32')
    distances, indices = faiss_index.search(scaled_vec, k * 3)

    results = df.iloc[indices[0]].copy()
    results['score'] = distances[0]

    if genre and genre != "Any":
        results = results[results['track_genre'].str.lower() == genre.lower()]
    if artist_filter:
        results = results[
            results['artists'].str.lower().str.contains(artist_filter.lower(), na=False)
        ]

    return results.head(k)

# ─── ReAct Agent ───────────────────────────────────────────────────────────────
def react_agent(user_query, retrieved_songs, filters, playlist_context=None):
    songs_json = retrieved_songs[['track_name', 'artists', 'track_genre',
                                   'danceability', 'energy', 'valence', 'tempo',
                                   'popularity']].head(30).to_dict('records')

    playlist_info = ""
    if playlist_context:
        playlist_info = f"\nUser's playlist taste profile:\n{json.dumps(playlist_context[:5], indent=2)}"

    system = f"""You are BeatBop, a music recommendation agent using ReAct (Reason + Act).

Your job:
1. REASON about what the user wants based on their query and filters
2. ACT by selecting the best songs from the retrieved list
3. OBSERVE the song features to ensure they match
4. Return final recommendations with brief reasoning

User filters applied: {json.dumps(filters)}
{playlist_info}

Retrieved songs from dataset (these are REAL songs — only recommend from this list):
{json.dumps(songs_json, indent=2)}

ReAct steps to follow:
Thought: analyze the query and what features matter most
Action: filter the list mentally based on valence/energy/tempo/genre
Observation: check if selected songs truly match
Final Answer: give top 5-7 recommendations with explanation

Format your final answer as:
🎵 **Track Name** - Artist (Genre) | Why: [1 line reason]
"""

    resp = client.chat.completions.create(
        model=st.session_state.get("model", "llama3-70b-8192"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_query}
        ],
        temperature=0.6,
        max_tokens=600
    )
    return resp.choices[0].message.content

# ─── Sidebar: Filters & Upload ─────────────────────────────────────────────────
with st.sidebar:
    st.header("🎛️ Filters")

    model = st.selectbox("Model", get_available_models(), key="model")

    st.subheader("🎭 Mood")
    mood = st.radio("Pick a mood", ["Any"] + list(MOOD_VECTORS.keys()), horizontal=False)

    st.subheader("⚡ Energy Level")
    energy = st.radio("Energy", ["Any"] + list(ENERGY_VECTORS.keys()), horizontal=False)

    st.subheader("🎸 Genre")
    genres = ["Any"] + sorted(df['track_genre'].unique().tolist())
    genre = st.selectbox("Genre", genres)

    st.subheader("🎤 Artist (optional)")
    artist_filter = st.text_input("Filter by artist name")

    st.divider()

    st.subheader("📄 Upload Your Playlist (PDF)")
    pdf_file = st.file_uploader("Upload playlist PDF", type=["pdf"])

    playlist_songs = None
    playlist_vec = None

    if pdf_file:
        with st.spinner("Reading your playlist..."):
            text = extract_songs_from_pdf(pdf_file)
            playlist_songs = parse_playlist_with_llm(text)
            if playlist_songs:
                playlist_vec = get_playlist_feature_vector(playlist_songs)
                st.success(f"✅ Found {len(playlist_songs)} songs in your playlist")
                with st.expander("Detected songs"):
                    for s in playlist_songs[:10]:
                        st.write(f"• {s['song']} — {s['artist']}")
            else:
                st.warning("Couldn't parse songs from PDF. Try a text-based PDF.")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ─── Chat UI ───────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show active filters
active = []
if mood != "Any": active.append(f"😊 {mood}")
if energy != "Any": active.append(f"⚡ {energy}")
if genre != "Any": active.append(f"🎸 {genre}")
if artist_filter: active.append(f"🎤 {artist_filter}")
if playlist_vec is not None: active.append("📄 Playlist loaded")

if active:
    st.info("Active filters: " + " · ".join(active))

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("🎤 Tell me what you're in the mood for..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Finding your perfect songs..."):
            # Build query vector from filters + playlist
            query_vec = build_query_vector(
                mood=mood if mood != "Any" else None,
                energy=energy if energy != "Any" else None,
                genre=genre if genre != "Any" else None,
                playlist_vec=playlist_vec
            )

            # RAG: retrieve from FAISS
            retrieved = retrieve_songs(
                query_vec,
                genre=genre,
                artist_filter=artist_filter if artist_filter else None
            )

            # ReAct: LLM reasons over retrieved songs
            filters_used = {
                "mood": mood, "energy": energy,
                "genre": genre, "artist": artist_filter
            }
            response = react_agent(
                prompt, retrieved, filters_used,
                playlist_context=playlist_songs
            )

            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})