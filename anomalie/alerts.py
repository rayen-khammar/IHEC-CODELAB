# alerts.py

import logging
from datetime import datetime


# =========================
# Logger configuration
# =========================
logging.basicConfig(
    filename="alerts.log",
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================
# Alert function
# =========================
def trigger_alerts(df):
    """
    Log and display alerts for detected anomalies.
    Input:
        df : DataFrame containing detected anomalies
             (output of anomaly_detector)
    """

    anomalies = df[df["is_anomaly"]]

    for _, row in anomalies.iterrows():
        message = (
            f"ANOMALY | "
            f"CODE={row['CODE']} | "
            f"TIME={row['SEANCE']} | "
            f"Volume={row['volume_anomaly']} | "
            f"Price={row['price_anomaly']} | "
            f"Pattern={row['pattern_anomaly']}"
        )

        # Log anomaly
        logging.warning(message)

        # Console alert (works everywhere)
        print(f"⚠️ {message}")
