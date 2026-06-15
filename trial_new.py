import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
import io
import json
import re
import base64
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

# ─── Similarity graph with top 50 given by FAISS ───────────────────────────────────────
def graph_rerank(candidates_df, query_vec, top_k=10, sim_threshold=0.85):
    """
    Re-rank FAISS candidates using a similarity graph to improve diversity.
    Penalizes songs that are too similar to already-selected songs.
    """
    # Get normalized feature vectors for candidates
    cand_features = scaler.transform(candidates_df[FEATURE_COLS].values).astype('float32')
    query_norm = query_vec.flatten()

    # Compute cosine similarity of each candidate to the query
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    query_scores = np.array([cosine_sim(f, query_norm) for f in cand_features])

    selected_indices = []
    selected_vectors = []

    for _ in range(top_k):
        best_score = -1
        best_idx = -1

        for i, (score, vec) in enumerate(zip(query_scores, cand_features)):
            if i in selected_indices:
                continue

            # Penalize if too similar to already selected songs
            if selected_vectors:
                max_sim_to_selected = max(cosine_sim(vec, s) for s in selected_vectors)
                if max_sim_to_selected > sim_threshold:
                    score *= (1 - max_sim_to_selected)  # diversity penalty

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx == -1:
            break

        selected_indices.append(best_idx)
        selected_vectors.append(cand_features[best_idx])

    return candidates_df.iloc[selected_indices].reset_index(drop=True)
# ─── Screenshot → Songs via Groq Vision ───────────────────────────────────────
def extract_tracks_from_screenshot(image_bytes: bytes, media_type: str):
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{b64}"},
                },
                {
                    "type": "text",
                    "text": (
                        "This is a screenshot of a music playlist (e.g. Spotify, Apple Music, YouTube Music). "
                        "Extract every visible track. Return ONLY a valid JSON array — no markdown, no explanation. "
                        "Each object: {\"song\": \"track name\", \"artist\": \"artist name\"}. "
                        "If artist is not visible use an empty string. Be thorough — extract ALL tracks visible."
                    ),
                },
            ],
        }],
        temperature=0.1,
        max_tokens=2000,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)

# ─── Playlist → Feature Vector ─────────────────────────────────────────────────
def get_taste_embedding(songs):
    """
    Builds a weighted taste embedding from playlist songs.
    Weights by popularity — more popular tracks = stronger preference signal.
    Falls back to equal weights if popularity is unavailable.
    """
    vectors = []
    weights = []

    for item in songs:
        song_name = item['song'].lower().strip()
        artist_name = item['artist'].lower().strip()

        name_match = df[df['track_name'].str.lower().str.contains(song_name, na=False, regex=False)]

        if artist_name and not name_match.empty:
            artist_match = name_match[name_match['artists'].str.lower().str.contains(artist_name, na=False, regex=False)]
            match = artist_match if not artist_match.empty else name_match
        else:
            match = name_match

        if not match.empty:
            row = match.iloc[0]
            vectors.append(row[FEATURE_COLS].values)
            weights.append(float(row.get('popularity', 50)) + 1)  # +1 to avoid zero weight

    if not vectors:
        return None

    vectors = np.array(vectors, dtype='float32')
    weights = np.array(weights, dtype='float32')
    weights /= weights.sum()  # normalize to sum to 1

    taste_vec = np.average(vectors, axis=0, weights=weights).astype('float32')
    return taste_vec

# ─── Filter → Feature Vector Mapping ──────────────────────────────────────────
MOOD_VECTORS = {
    "Happy":     dict(valence=0.8, energy=0.7, danceability=0.7),
    "Sad":       dict(valence=0.2, energy=0.3, acousticness=0.7),
    "Party":     dict(danceability=0.9, energy=0.9, valence=0.7),
    "Chill":     dict(energy=0.3, acousticness=0.6, tempo=0.3),
    "Angry":     dict(energy=0.95, valence=0.1, speechiness=0.7, danceability=0.4),
    "Romantic":  dict(valence=0.6, acousticness=0.6, energy=0.4),
    "Uplifting": dict(valence=0.85, energy=0.75, danceability=0.65),
    "Focus":     dict(instrumentalness=0.7, energy=0.4, speechiness=0.1),
}

ENERGY_VECTORS = {
    "High Beat":  dict(tempo=0.9, energy=0.9, danceability=0.8),
    "Medium":     dict(tempo=0.5, energy=0.5),
    "Low / Calm": dict(tempo=0.2, energy=0.2, acousticness=0.7),
}

def build_intent_vector(mood=None, energy=None):
    """Short-term intent from explicit user input."""
    base = {col: 0.5 for col in FEATURE_COLS}
    if mood and mood in MOOD_VECTORS:
        base.update(MOOD_VECTORS[mood])
    if energy and energy in ENERGY_VECTORS:
        base.update(ENERGY_VECTORS[energy])
    return np.array([base[c] for c in FEATURE_COLS], dtype='float32')

def blend_vectors(taste_vec=None, intent_vec=None, has_explicit_intent=False):
    """
    Blend long-term taste with short-term intent.
    
    - No playlist, no intent   → neutral vector (fallback)
    - Playlist only            → 100% taste
    - Intent only              → 100% intent  
    - Both                     → 60% intent, 40% taste (intent takes lead when explicit)
    """
    if taste_vec is None and not has_explicit_intent:
        # Cold start: return neutral
        return np.array([0.5] * len(FEATURE_COLS), dtype='float32').reshape(1, -1)

    if taste_vec is not None and not has_explicit_intent:
        return taste_vec.reshape(1, -1)

    if taste_vec is None and has_explicit_intent:
        return intent_vec.reshape(1, -1)

    # Both available — intent leads
    alpha = 0.6  # weight for intent
    blended = alpha * intent_vec + (1 - alpha) * taste_vec
    return blended.reshape(1, -1)

# ─── RAG Retrieval ─────────────────────────────────────────────────────────────
def retrieve_songs(query_vec, genre=None, artist_filter=None, k=50):
    scaled_vec = scaler.transform(query_vec).astype('float32')
    distances, indices = faiss_index.search(scaled_vec, k * 3)

    results = df.iloc[indices[0]].copy()
    results['score'] = distances[0]

    if genre and genre != "Any":
        results = results[results['track_genre'].str.lower() == genre.lower()]
    if artist_filter:
        results = results[results['artists'].str.lower().str.contains(artist_filter.lower(), na=False, regex=False)]

    results = results.head(k)
    results = graph_rerank(results, query_vec, top_k=15)  # ← add this line
    return results

# ─── ReAct Agent ───────────────────────────────────────────────────────────────
def react_agent(user_query, retrieved_songs, filters, playlist_context=None):
    songs_json = retrieved_songs[['track_name', 'artists', 'track_genre',
                                   'danceability', 'energy', 'valence', 'tempo',
                                   'popularity']].head(30).to_dict('records')

    playlist_info = ""
    if playlist_context:
        playlist_info = f"""
            User's long-term taste was extracted from their playlist ({len(playlist_context)} songs).
            This was blended with their current mood/energy intent.
            Sample playlist songs: {json.dumps(playlist_context[:5], indent=2)}
            """

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

# ─── Multi-Chat Session State Init ────────────────────────────────────────────
# Structure: st.session_state.chats = { "Chat 1": { "messages": [], "playlist_songs": [] } }
if "chats" not in st.session_state:
    st.session_state.chats = {"Chat 1": {"messages": [], "playlist_songs": None}}
if "active_chat" not in st.session_state:
    st.session_state.active_chat = "Chat 1"

def current_chat():
    return st.session_state.chats[st.session_state.active_chat]

# ─── Sidebar ──────────────────────────────────────────────────────────────────
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

    # ── Screenshot Upload ──────────────────────────────────────────────────────
    st.subheader("📸 Upload Playlist Screenshot")
    uploaded_img = st.file_uploader("Upload a screenshot of any playlist",
                                    type=["png", "jpg", "jpeg", "webp"])

    playlist_songs = current_chat().get("playlist_songs")
    playlist_vec   = get_taste_embedding(playlist_songs) if playlist_songs else None

    if uploaded_img and st.button("Extract Tracks"):
        with st.spinner("Reading screenshot with Groq Vision..."):
            try:
                mt = uploaded_img.type or "image/png"
                songs = extract_tracks_from_screenshot(uploaded_img.read(), mt)
                current_chat()["playlist_songs"] = songs
                playlist_songs = songs
                playlist_vec   = get_taste_embedding(songs)
                st.success(f"✅ Extracted {len(songs)} tracks")
                with st.expander("Detected songs"):
                    for s in songs[:15]:
                        st.write(f"• {s['song']} — {s['artist']}")
            except Exception as e:
                st.error(f"❌ Could not extract tracks: {e}")

    elif playlist_songs:
        st.success(f"✅ {len(playlist_songs)} tracks loaded")
        with st.expander("Loaded songs"):
            for s in playlist_songs[:15]:
                st.write(f"• {s['song']} — {s['artist']}")

    st.divider()

    # ── Multi-Chat Manager ─────────────────────────────────────────────────────
    st.subheader("💬 Chat Sessions")

    chat_names = list(st.session_state.chats.keys())
    selected = st.radio("Switch chat", chat_names, 
                        index=chat_names.index(st.session_state.active_chat),
                        key="chat_selector")
    if selected != st.session_state.active_chat:
        st.session_state.active_chat = selected
        st.rerun()

    # New chat
    col1, col2 = st.columns([3, 1])
    with col1:
        new_name = st.text_input("New chat name", placeholder="e.g. Workout Mix",
                                  label_visibility="collapsed")
    with col2:
        if st.button("➕") and new_name.strip():
            if new_name.strip() not in st.session_state.chats:
                st.session_state.chats[new_name.strip()] = {"messages": [], "playlist_songs": None}
                st.session_state.active_chat = new_name.strip()
                st.rerun()

    # Delete current chat
    if len(st.session_state.chats) > 1:
        if st.button(f"🗑️ Delete '{st.session_state.active_chat}'"):
            del st.session_state.chats[st.session_state.active_chat]
            st.session_state.active_chat = list(st.session_state.chats.keys())[0]
            st.rerun()

    if st.button("🗑️ Clear Current Chat"):
        current_chat()["messages"] = []
        st.rerun()

# ─── Chat UI ──────────────────────────────────────────────────────────────────
st.subheader(f"💬 {st.session_state.active_chat}")

# Show active filters
active = []
if mood != "Any":            active.append(f"😊 {mood}")
if energy != "Any":          active.append(f"⚡ {energy}")
if genre != "Any":           active.append(f"🎸 {genre}")
if artist_filter:            active.append(f"🎤 {artist_filter}")
if playlist_vec is not None: active.append("📸 Playlist loaded")

if active:
    st.info("Active filters: " + " · ".join(active))

messages = current_chat()["messages"]

for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("🎤 Tell me what you're in the mood for..."):
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Finding your perfect songs..."):
            has_intent = (mood != "Any") or (energy != "Any")
            intent_vec = build_intent_vector(
                mood=mood if mood != "Any" else None,
                energy=energy if energy != "Any" else None
            )
            query_vec = blend_vectors(
                taste_vec=playlist_vec,
                intent_vec=intent_vec,
                has_explicit_intent=has_intent
            )
            retrieved = retrieve_songs(
                query_vec,
                genre=genre,
                artist_filter=artist_filter if artist_filter else None
            )
            filters_used = {
                "mood": mood, "energy": energy,
                "genre": genre, "artist": artist_filter
            }
            response = react_agent(
                prompt, retrieved, filters_used,
                playlist_context=playlist_songs
            )
            st.markdown(response)

    messages.append({"role": "assistant", "content": response})