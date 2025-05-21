from dotenv import load_dotenv
import streamlit as st
import requests
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
import torch

load_dotenv()
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
SPOTIFY_TOKEN_URL = 'https://accounts.spotify.com/api/token'
SPOTIFY_SEARCH_URL = 'https://api.spotify.com/v1/search'

@st.cache_resource
def load_emotion_model():
    model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
    tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
    return model, tokenizer

def detect_mood(text):
    model, tokenizer = load_emotion_model()
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    labels = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
        "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
        "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
        "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
    ]
    return labels[predicted_class_id]

def refine_mood(label):
    mapping = {
        "joy": "happy", "love": "happy", "gratitude": "happy", "pride": "happy", "admiration": "happy",
        "amusement": "excited", "surprise": "excited",
        "anger": "angry", "annoyance": "angry", "disapproval": "angry",
        "fear": "anxious", "nervousness": "anxious",
        "remorse": "upset", "disgust": "upset",
        "grief": "depressed", "sadness": "depressed", "disappointment": "depressed",
        "neutral": "calm"
    }
    return mapping.get(label, "calm")

@st.cache_resource
def load_lyrics_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer

def generate_lyrics(prompt, model, tokenizer):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2,
                            pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def get_spotify_token():
    auth_response = requests.post(
        SPOTIFY_TOKEN_URL,
        {
            'grant_type': 'client_credentials',
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
        },
    )
    if auth_response.status_code != 200:
        st.error(f"Spotify auth failed: {auth_response.text}")
        return None
    return auth_response.json().get('access_token')

def get_spotify_tracks_filtered(mood, token):
    if not token:
        return []

    genre_map = {
        "happy": ["pop", "dance", "indie-pop", "k-pop"],
        "depressed": ["acoustic", "sad", "piano", "ambient"],
        "angry": ["metal", "hard-rock", "punk", "rock"],
        "anxious": ["ambient", "chill", "classical", "lofi"],
        "excited": ["edm", "electro", "dance", "pop"],
        "calm": ["lofi", "chill", "jazz", "classical"],
        "upset": ["sad", "piano", "acoustic", "singer-songwriter"]
    }

    genres = genre_map.get(mood, ["pop"])
    headers = {'Authorization': f'Bearer {token}'}
    tracks = []

    for genre in genres:
        params = {'q': f"{mood} {genre}", 'type': 'track', 'limit': 4}
        response = requests.get(SPOTIFY_SEARCH_URL, headers=headers, params=params)
        if response.status_code == 200:
            items = response.json().get('tracks', {}).get('items', [])
            for item in items:
                tracks.append({
                    'name': item['name'],
                    'url': item['external_urls']['spotify'],
                    'preview_url': item['preview_url'],
                    'image': item['album']['images'][0]['url'] if item['album']['images'] else None,
                    'artist': item['artists'][0]['name']
                })

    seen = set()
    unique_tracks = []
    for track in tracks:
        if track['name'] not in seen:
            unique_tracks.append(track)
            seen.add(track['name'])

    return unique_tracks[:12]

def save_favorite(track, mood):
    data = {"mood": mood, "track": track}
    if os.path.exists("favorites.json"):
        with open("favorites.json", "r") as f:
            favs = json.load(f)
    else:
        favs = []
    favs.append(data)
    with open("favorites.json", "w") as f:
        json.dump(favs, f, indent=4)

def show_favorites():
    if os.path.exists("favorites.json"):
        with open("favorites.json", "r") as f:
            favs = json.load(f)
        st.write("### 📚 Your Favorite Tracks")
        for entry in favs:
            track = entry["track"]
            mood = entry["mood"]
            st.markdown(f"**Mood:** {mood}")
            st.markdown(f"🎵 [{track['name']}]({track['url']}) by *{track['artist']}*")
            if track.get("image"):
                st.image(track["image"], width=100)
    else:
        st.info("No favorites saved yet.")

st.set_page_config(page_title="Mood-Based Music Recommender", layout="centered")
st.title("🎵 Mood-Based Music Recommender")

user_input = st.text_area("How are you feeling today?", placeholder="e.g. I feel like I want to cry")

if user_input:
    raw_emotion = detect_mood(user_input)
    mood = refine_mood(raw_emotion)
    st.success(f"Detected Mood: **{mood.capitalize()}** ({raw_emotion})")

    token = get_spotify_token()
    st.info("Fetching mood-based tracks...")

    tracks = get_spotify_tracks_filtered(mood, token)

    if tracks:
        st.write("### 🎧 Matching Spotify Tracks")
        cols_per_row = 4
        rows = 3
        total = min(len(tracks), cols_per_row * rows)
        for row in range(rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                idx = row * cols_per_row + col_idx
                if idx >= total:
                    break
                track = tracks[idx]
                with cols[col_idx]:
                    if track['image']:
                        st.image(track['image'], use_container_width=True)
                    st.markdown(f"**{track['name']}**  \n*by {track['artist']}*")
                    st.markdown(f"[🔗 Open in Spotify]({track['url']})")
                    if track['preview_url']:
                        st.audio(track['preview_url'], format="audio/mp3")
                    if st.button("💾 Add to Favorites", key=f"fav_{idx}"):
                        save_favorite(track, mood)
                        st.success("Added to favorites!")
    else:
        st.warning("No tracks found for this mood.")

    st.write("### 🎼 AI-Generated Lyrics")
    model, tokenizer = load_lyrics_model()
    prompt = f"A song about feeling {mood}: "
    lyrics = generate_lyrics(prompt, model, tokenizer)
    st.code(lyrics, language="text")

    if st.button("📖 Show My Favorites"):
        show_favorites()
