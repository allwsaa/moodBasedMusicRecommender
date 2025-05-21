# 🎵 Mood-Based Music Recommender

This Streamlit app detects a user’s mood from text input and recommends Spotify tracks that match the emotional tone. It also generates AI-written song lyrics using GPT-2 and displays tracks in a 4×3 grid with cover art, audio previews, and saving options.

---

## 🚀 Features

- 🧠 **Emotion Detection** using `roberta-base-go_emotions` model.
- 🎧 **Spotify Track Recommendations** tailored to emotional input and genre.
- 🤖 **AI-Generated Lyrics** from GPT-2 based on detected mood.
- 💾 **Favorites Saving** to `favorites.json`.
- 🎨 **Responsive Grid Layout**: tracks displayed in a 4×3 grid with images and audio.
- 🔊 Built-in preview player for each track.

---

## 🛠️ Technologies Used

| Feature                  | Technology                                |
|--------------------------|--------------------------------------------|
| Web App UI               | Streamlit                                 |
| Emotion Classification   | Transformers – `SamLowe/roberta-base-go_emotions` |
| Text Generation          | Transformers – `gpt2`                     |
| Music Search             | Spotify API (Client Credentials Flow)     |
| Language                 | Python 3.10+                              |

---

## 🎶 Mood-to-Genre Mapping

Used to filter search queries and improve relevance:

```json
{
  "happy": ["pop", "dance", "indie-pop", "k-pop"],
  "depressed": ["acoustic", "sad", "piano", "ambient"],
  "angry": ["metal", "hard-rock", "punk", "rock"],
  "anxious": ["ambient", "chill", "classical", "lofi"],
  "excited": ["edm", "electro", "dance", "pop"],
  "calm": ["lofi", "chill", "jazz", "classical"],
  "upset": ["sad", "piano", "acoustic", "singer-songwriter"]
}
```

##File Structure
```bash
.
├── app.py               # Main Streamlit app
├── favorites.json       # Saved favorite tracks
├── .env                 # Spotify API credentials
├── README.md            # This file
└── .venv/               # Virtual environment (optional)
```

