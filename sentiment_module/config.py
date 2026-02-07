import os

# =========================
# OpenAI configuration
# =========================

# The API key must be set as an environment variable
# export OPENAI_API_KEY="your_key_here"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise EnvironmentError(
        "OPENAI_API_KEY environment variable is not set"
    )

# =========================
# Sentiment thresholds
# =========================

# These thresholds are project-defined and explainable
SENTIMENT_THRESHOLDS = {
    "negative": -0.3,
    "positive": 0.3
}
  


"""
Commande pour lancer le module

Linux / Mac :

export OPENAI_API_KEY="sk-xxxxx"
python sentiment_analyzer.py


Windows (PowerShell) :

setx OPENAI_API_KEY "sk-xxxxx"
python sentiment_analyzer.py

"""
