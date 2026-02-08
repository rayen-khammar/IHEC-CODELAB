# 🧠 Intelligent Trading Assistant for BVMT

A **price-centric, risk-aware, and explainable AI system** designed to assist investors on the Tunisian Stock Exchange (BVMT).

Our solution predicts short-term price movements, assesses risk dynamically, personalizes decisions based on user behavior, and explains every recommendation in a transparent and human-understandable way.

---

## 🚀 Core Idea

**All investment decisions revolve around price.**

In our system:
- **Price prediction** is the central signal.
- **Risk** is quantified using prediction uncertainty and anomaly detection.
- **User behavior** shapes decisions through reinforcement learning.
- **Market sentiment** amplifies or dampens expected returns.
- **Explainability** ensures trust and transparency.

> **Price is the truth. Other signals are evidence.**

---

## 🧩 Key Features

- 📈 Short-term price prediction (1 to 5 days)
- ⚠️ Risk estimation using variance and anomalies
- 📰 Multilingual sentiment analysis (French / Arabic)
- 🧠 Personalized risk profiling via reinforcement learning
- 🔍 Anomaly detection (market + model drift)
- 💼 Portfolio simulation and future value projection
- 🧾 Explainable AI using signal decomposition and similar past cases

---

## 🔄 End-to-End Decision Pipeline

This pipeline shows **how raw data becomes an investment decision**.

### Pipeline Overview

1. **Market Data Ingestion**
   - BVMT OHLCV historical data
   - Near real-time scraped market values

2. **Feature Engineering**
   - Returns, volatility, technical indicators

3. **Price Prediction**
   - Next-day price forecast
   - Rolled forward to 5 days

4. **Uncertainty Estimation**
   - Prediction variance
   - Rolling prediction error

5. **Sentiment Analysis**
   - Financial news scraping (FR / AR)
   - Sentiment score per stock

6. **Anomaly Detection**
   - Market anomalies (volume, price spikes)
   - Model anomalies (prediction vs reality divergence)

7. **User Risk Profiling**
   - Reinforcement learning estimates user risk tolerance

8. **Decision Engine**
   - Combines return, risk, sentiment, and user profile
   - Outputs BUY / HOLD / SELL with confidence

9. **Portfolio Simulation**
   - Projects portfolio value for the next 1–5 days

10. **Explainability**
    - Explains decisions using signal contributions
    - References similar historical situations

---

## 🧠 Decision Logic (Simplified)

The system optimizes a **mean–variance utility function**:




Where:
- `α` controls sentiment influence
- `λ(user)` is learned via reinforcement learning
- Anomalies increase risk and reduce confidence

---

## 🏗️ System Architecture

Our architecture is **modular, robust, and production-oriented**.

### Architecture Layers

### 1️⃣ Data Layer
- BVMT historical market data
- Live market scraping
- Financial news sources

### 2️⃣ Intelligence Layer
- Price prediction engine
- Risk & variance estimator
- Sentiment analysis (NLP)
- Anomaly detection
- Reinforcement learning (user risk)

### 3️⃣ Decision Layer
- Decision & utility engine
- Portfolio simulator

### 4️⃣ Explainability Layer
- XAI engine
- Similar historical cases
- Decision decomposition

### 5️⃣ Interface Layer
- Market dashboard
- Alerts & anomalies
- Portfolio visualization

Each layer is **independent**, allowing:
- Easy upgrades
- Fault tolerance
- Regulatory extensions (CMF use cases)

---

## 🔍 Explainable AI (XAI)

Every recommendation is explained using:

- 📊 Predicted return and uncertainty
- ⚠️ Risk factors (variance + anomaly signals)
- 📰 Sentiment influence
- 📚 Similar past situations and outcomes

Example explanation:
> *“In similar market conditions, the price increased in 4 out of 5 cases.  
Current sentiment is positive, and no abnormal market behavior was detected.”*

---

## 💼 Portfolio Simulation

Instead of only giving recommendations, the system:
- Simulates portfolio evolution
- Projects gains/losses over the next 5 days
- Helps users understand the impact **before acting**

---

## 🛡️ Robust by Design

Because modules were developed independently:
- All integrations are **defensive**
- Missing or unreliable signals fall back to safe defaults
- The system always runs end-to-end

> This reflects real-world financial system constraints.

---

## 🔮 Future Improvements

- Multi-asset portfolio optimization
- Continuous online learning
- Advanced uncertainty estimation
- Regulatory monitoring dashboards (CMF)
- Cross-user anonymized learning

---

## 🏁 Conclusion

This project demonstrates how **AI, finance, and explainability** can be combined into a **trustworthy and practical trading assistant**, tailored to the realities of the Tunisian market.

---

**Built for hackathon speed. Designed for real-world deployment.**







