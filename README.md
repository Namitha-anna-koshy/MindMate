# 💬 MindMate – Your Caring AI Chat Buddy

A calming, empathetic mental health assistant powered by **LLaMA3 (Groq)** and built with 🧠 **LangChain** + 🌐 **Streamlit**.

Perfect for anyone who needs a gentle nudge toward self-care — or just wants to vibe-chat with an AI that actually listens.

## Getting Started

### 1. Clone the repo

git clone https://github.com/Namitha-anna-koshy/MindMate.git
cd mindmate

### 2. Set up your Python environment
Make sure you have Python 3.10+ installed (and preferably a virtual environment):

pip install -r requirements.txt

### 🔐 Set Up Your API Key
This app uses Groq's API to run LLaMA3 models at warp speed ⚡.

### Steps to get your API key:

Go to https://console.groq.com/keys
Sign in or create a free account
Generate a new API key
Create a .env file in the root directory of the project
Paste this inside:

GROQ_API_KEY=your-super-secret-api-key-here
⚠️ Never commit your .env file to GitHub. Seriously. Don’t.

▶️ Run the App
Once everything's ready, launch the app:

streamlit run app.py

Your calming chatbot will be live at http://localhost:8501. Take a breath and say hi 👋

🧠 Features
💬 Chat with an emotionally intelligent AI
🌿 Suggests gentle self-care based on your mood
😌 Uses LLaMA3-70B (Groq) for calm, human-like responses
🔍 Detects basic emotional states
✨ Simple, aesthetic Streamlit UI

Built this for a hackathon.
Need improvements

