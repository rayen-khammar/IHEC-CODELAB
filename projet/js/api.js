import { API_BASE, FETCH_TIMEOUT_MS, ALLOW_MOCK_FALLBACK } from "./config.js";

function withTimeout(promise, ms) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  return { controller, wrapped: promise(controller.signal).finally(() => clearTimeout(timer)) };
}

async function fetchJSON(path) {
  const url = `${API_BASE}${path}`;

  const { wrapped } = withTimeout(async (signal) => {
    const res = await fetch(url, { signal, headers: { "Accept": "application/json" } });
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${path}`);
    return res.json();
  }, FETCH_TIMEOUT_MS);

  return wrapped;
}

/**
 * Expected endpoints (you can adapt paths in one place here):
 * - GET /market/overview
 * - GET /market/stock?symbol=XXX
 * - GET /market/alerts
 * - GET /user/profile (optional)
 */
export async function getMarketOverview() {
  try {
    return await fetchJSON("/market/overview");
  } catch (e) {
    if (!ALLOW_MOCK_FALLBACK) throw e;
    return mockOverview();
  }
}

export async function getStockDetails(symbol) {
  try {
    return await fetchJSON(`/market/stock?symbol=${encodeURIComponent(symbol)}`);
  } catch (e) {
    if (!ALLOW_MOCK_FALLBACK) throw e;
    return mockStock(symbol);
  }
}

export async function getAlerts() {
  try {
    return await fetchJSON("/market/alerts");
  } catch (e) {
    if (!ALLOW_MOCK_FALLBACK) throw e;
    return mockAlerts();
  }
}

export async function getUserProfile() {
  try {
    return await fetchJSON("/user/profile");
  } catch (e) {
    if (!ALLOW_MOCK_FALLBACK) throw e;
    return mockProfile();
  }
}

/* ------------------ MOCK FALLBACKS ------------------ */

function mockOverview() {
  const now = new Date().toISOString();
  return {
    last_updated: now,
    indices: [
      { name: "TUNINDEX", value: 9134.22, change_percent: 1.42, trend_24h: [9050, 9062, 9071, 9090, 9102, 9120, 9134] },
      { name: "TUNINDEX20", value: 4021.55, change_percent: -0.38, trend_24h: [4040, 4036, 4030, 4028, 4025, 4023, 4021.55] }
    ],
    sentiment: { score: 0.18, label: "Slightly Bullish", confidence: 0.72, sources_count: 126 },
    top_gainers: [
      { symbol: "SFBT", name: "SFBT", last: 12.40, change_percent: 4.12, volume: 245000 },
      { symbol: "BIAT", name: "BIAT", last: 98.20, change_percent: 2.30, volume: 51000 },
      { symbol: "TELNET", name: "TELNET", last: 6.35, change_percent: 1.95, volume: 88000 },
      { symbol: "ATTIJ", name: "ATTIJ", last: 46.80, change_percent: 1.64, volume: 74000 },
      { symbol: "DELICE", name: "DELICE", last: 9.12, change_percent: 1.10, volume: 112000 }
    ],
    top_losers: [
      { symbol: "POULINA", name: "PGH", last: 7.05, change_percent: -3.22, volume: 93000 },
      { symbol: "TPR", name: "TPR", last: 4.88, change_percent: -2.10, volume: 61000 },
      { symbol: "SOTUMAG", name: "SOTUMAG", last: 2.31, change_percent: -1.80, volume: 28000 },
      { symbol: "ENNAKL", name: "ENNAKL", last: 10.90, change_percent: -1.15, volume: 39000 },
      { symbol: "UIB", name: "UIB", last: 22.40, change_percent: -0.92, volume: 17000 }
    ],
    alerts_recent: [
      { id: "a1", ts: now, symbol: "SFBT", type: "VOLUME_SPIKE", severity: "HIGH", title: "Pic de volume détecté (+320%)" },
      { id: "a2", ts: now, symbol: "DELICE", type: "PRICE_MOVE_NO_NEWS", severity: "MEDIUM", title: "Variation anormale sans news" }
    ]
  };
}

function mockStock(symbol) {
  const now = new Date();
  const hist = [];
  const pred = [];
  let price = 10 + Math.random() * 5;

  for (let i = 30; i >= 1; i--) {
    price += (Math.random() - 0.5) * 0.3;
    const d = new Date(now);
    d.setDate(now.getDate() - i);
    hist.push({ date: d.toISOString().slice(0, 10), close: +price.toFixed(2) });
  }

  let p2 = price;
  for (let i = 1; i <= 5; i++) {
    p2 += (Math.random() - 0.45) * 0.4;
    const d = new Date(now);
    d.setDate(now.getDate() + i);
    pred.push({
      date: d.toISOString().slice(0, 10),
      close_pred: +p2.toFixed(2),
      ci_low: +(p2 - 0.6).toFixed(2),
      ci_high: +(p2 + 0.6).toFixed(2),
    });
  }

  return {
    symbol,
    name: symbol,
    last_updated: now.toISOString(),
    history: hist,
    forecast_5d: pred,
    sentiment: { series: hist.slice(-14).map((x, i) => ({ date: x.date, score: +(Math.sin(i / 3) * 0.4).toFixed(2) })) },
    indicators: { rsi: 57.2, macd: 0.12, signal: 0.08 },
    recommendation: {
      action: "HOLD",
      confidence: 0.63,
      explanation: "Prévision légèrement haussière mais volatilité moyenne; aucun signal d’anomalie fort détecté."
    }
  };
}

function mockAlerts() {
  const now = new Date().toISOString();
  return {
    last_updated: now,
    items: [
      { id: "a1", ts: now, symbol: "SFBT", type: "VOLUME_SPIKE", severity: "HIGH", title: "Pic de volume détecté (+320%)" },
      { id: "a2", ts: now, symbol: "DELICE", type: "PRICE_MOVE_NO_NEWS", severity: "MEDIUM", title: "Variation anormale sans news" },
      { id: "a3", ts: now, symbol: "BIAT", type: "PRICE_SPIKE", severity: "LOW", title: "Variation de prix inhabituelle" }
    ]
  };
}

function mockProfile() {
  return {
    name: "Guest",
    risk_profile: "Moderate",
    favorites: ["SFBT", "BIAT", "DELICE"]
  };
}
