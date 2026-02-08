import os
from dotenv import load_dotenv

load_dotenv()

# 🔥 Mets ici la page BVMT réelle (page avec le tableau cotation)
BVMT_URL = "https://www.bvmt.com.tn/"

# fréquence du refresh (secondes)
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "15"))

# base sqlite
DB_URL = os.getenv("DB_URL", "sqlite:///bvmt_live.db")
