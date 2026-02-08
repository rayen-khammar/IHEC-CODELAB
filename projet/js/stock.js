/* ================================
   TUNISIAN COMPANIES LIST (20+)
   Later → Can come from API
================================ */

const tunisianCompanies = [
"SFBT",
"BIAT",
"ATB",
"BH BANK",
"STB",
"AMEN BANK",
"UBCI",
"DELICE",
"POULINA",
"CARTHAGE CEMENT",
"TPR",
"ASSAD",
"ENNAKL",
"MPBS",
"SOTUVER",
"TELNET",
"SIPHAT",
"MONOPRIX",
"SMART TUNISIE",
"OFFICE PLAST"
];

/* ================================
   LOAD COMPANIES INTO DROPDOWN
================================ */

const select = document.getElementById("companySelect");

tunisianCompanies.forEach(c=>{
  const opt=document.createElement("option");
  opt.value=c;
  opt.textContent=c;
  select.appendChild(opt);
});

/* ================================
   DEMO DATA (REMOVE WHEN API READY)
================================ */

function loadFakeData(){

  /* 
  FUTURE API EXAMPLE:

  fetch("/api/stock?symbol="+symbol)
  .then(r=>r.json())
  .then(data=>{
      updateUI(data)
  })
  */

  document.getElementById("pred1d").innerText = "—";
  document.getElementById("pred5d").innerText = "—";

  document.getElementById("predictionReason").innerText =
  "Strong earnings + positive macro outlook";

  document.getElementById("recommendation").innerText = "—";
  document.getElementById("recommendReason").innerText =
  "Momentum breakout + institutional buying";

  document.getElementById("sentimentValue").innerText = "—";
  document.getElementById("sentimentReason").innerText =
  "Positive news + social media sentiment";

  document.getElementById("rsiVal").innerText = "—";
  document.getElementById("macdVal").innerText = "—";
  document.getElementById("signalVal").innerText = "—";

  document.getElementById("indicatorReason").innerText =
  "RSI above 60 + MACD crossover";
}

/* ================================
   ON COMPANY CHANGE
================================ */

select.addEventListener("change",()=>{
  if (typeof window.renderStockDemo === "function") {
    window.renderStockDemo();
  } else {
    loadFakeData();
  }
});

/* INIT */
loadFakeData();

const modal = document.getElementById("buyModal");
const confirmBtn = document.getElementById("confirmBuyBtn");

let selectedPrice = 0;

/* =========================
   OPEN BUY MODAL
========================= */

document.getElementById("buyBtn").onclick = ()=>{

  const symbol = document.getElementById("companySelect").value;

  // Extract price from UI
  const priceText = document.getElementById("pred1d").innerText;
  selectedPrice = parseFloat(priceText);

  document.getElementById("tradeSymbol").innerText = symbol;
  document.getElementById("tradePrice").innerText = priceText;

  modal.classList.remove("hidden");
};

document.addEventListener("DOMContentLoaded", () => {

  /* ===============================
     1) SAFE HELPERS
  =============================== */

  const $ = (id) => document.getElementById(id);

  function money(n){
    return `${Number(n).toFixed(3)} TND`;
  }

  function readJSON(key, fallback){
    try{
      const v = localStorage.getItem(key);
      return v ? JSON.parse(v) : fallback;
    }catch{
      return fallback;
    }
  }

  function writeJSON(key, value){
    localStorage.setItem(key, JSON.stringify(value));
  }

  /* ===============================
     2) PORTFOLIO STORAGE (GLOBAL)
     - funds + holdings
  =============================== */

  const PORTFOLIO_KEY = "bvmt_portfolio";

  function getPortfolio(){
    const p = readJSON(PORTFOLIO_KEY, null);
    if(p && typeof p.funds === "number" && Array.isArray(p.holdings)) return p;

    // default
    const init = { funds: 5000, holdings: [] };
    writeJSON(PORTFOLIO_KEY, init);
    return init;
  }

  function savePortfolio(p){
    writeJSON(PORTFOLIO_KEY, p);
  }

  function buy(symbol, price, qty){
    const p = getPortfolio();
    const cost = price * qty;

    if(!Number.isFinite(price) || price <= 0){
      return { ok:false, msg:"Invalid price." };
    }
    if(!Number.isInteger(qty) || qty <= 0){
      return { ok:false, msg:"Quantity must be a positive integer." };
    }
    if(p.funds < cost){
      return { ok:false, msg:`Insufficient funds. Need ${money(cost)}, you have ${money(p.funds)}.` };
    }

    // Deduct funds
    p.funds = Number((p.funds - cost).toFixed(3));

    // Add holding (merge if same symbol)
    const existing = p.holdings.find(h => h.symbol === symbol);
    if(existing){
      // Weighted average price
      const totalQty = existing.qty + qty;
      existing.avgPrice = ((existing.avgPrice * existing.qty) + (price * qty)) / totalQty;
      existing.qty = totalQty;
    }else{
      p.holdings.push({ symbol, qty, avgPrice: price });
    }

    savePortfolio(p);
    return { ok:true, msg:`Bought ${qty} ${symbol} for ${money(cost)}.` };
  }

  /* ===============================
     3) COMPANY LIST
  =============================== */

  const tunisianCompanies = [
    "SFBT","BIAT","ATB","BH BANK","STB","AMEN BANK","UBCI","DELICE","POULINA",
    "CARTHAGE CEMENT","TPR","ASSAD","ENNAKL","MPBS","SOTUVER","TELNET","SIPHAT",
    "MONOPRIX","SAH","EURO-CYCLES","ADWYA","ONE TECH","CITY CARS","ARTES",
  ];

  const select = $("companySelect");
  if(select){
    select.innerHTML = "";
    tunisianCompanies.forEach(sym=>{
      const opt = document.createElement("option");
      opt.value = sym;
      opt.textContent = sym;
      select.appendChild(opt);
    });
  }

  /* ===============================
     4) DEMO "API" DATA
     Replace later with real fetch()
  =============================== */

  function fakeApiForSymbol(symbol){
    const base = {
      "SFBT": 12.300,
      "BIAT": 98.000,
      "DELICE": 9.100,
      "TPR": 6.800,
      "TELNET": 7.400,
      "ATTIJARI BANK": 30.400,
    }[symbol] || 10.000;

    return {
      pred1d: base * 1.012,
      pred5d: base * 1.048,
      predictionReason: "Earnings momentum + positive sector flows (demo placeholder).",
      recommendation: "BUY",
      recommendReason: "Technical breakout + improving liquidity (demo placeholder).",
      sentiment: "Bullish",
      sentimentReason: "News + market signals aggregated (demo placeholder).",
      indicators: { rsi: 62, macd: "Positive", signal: "Buy" },
      indicatorReason: "RSI > 60 and MACD crossover (demo placeholder)."
    };
  }

  function renderStock(){
    const symbol = select ? select.value : "SFBT";
    const data = fakeApiForSymbol(symbol);

    if($("pred1d")) $("pred1d").innerText = money(data.pred1d);
    if($("pred5d")) $("pred5d").innerText = money(data.pred5d);

    const pr = $("predictionReason");
    if(pr) pr.innerText = data.predictionReason;

    const rr = $("recommendReason");
    if(rr) rr.innerText = data.recommendReason;

    const sr = $("sentimentReason");
    if(sr) sr.innerText = data.sentimentReason;

    const ir = $("indicatorReason");
    if(ir) ir.innerText = data.indicatorReason;

    if($("recommendation")) $("recommendation").innerText = data.recommendation;
    if($("sentimentValue")) $("sentimentValue").innerText = data.sentiment;

    if($("rsiVal")) $("rsiVal").innerText = data.indicators.rsi;
    if($("macdVal")) $("macdVal").innerText = data.indicators.macd;
    if($("signalVal")) $("signalVal").innerText = data.indicators.signal;
  }

  if(select){
    select.addEventListener("change", renderStock);
  }

  /* ===============================
     5) BUY MODAL FLOW
  =============================== */

  const buyBtn = $("buyBtn");
  const modal = $("buyModal");
  const closeBtn = $("closeTradeBtn");
  const cancelBtn = $("cancelTradeBtn");
  const confirmBtn = $("confirmBuyBtn");

  function openModal(){
    if(!modal) return;

    const symbol = select ? select.value : "SFBT";
    const priceText = $("pred1d")?.innerText || "0";
    const price = parseFloat(priceText); // works because money() starts with number

    const p = getPortfolio();

    $("tradeSymbol").innerText = symbol;
    $("tradePrice").innerText = priceText;
    $("tradeFunds").innerText = money(p.funds);
    $("tradeQty").value = "";
    $("tradeMsg").innerText = "";

    modal.classList.remove("hidden");
  }

  function closeModal(){
    if(!modal) return;
    modal.classList.add("hidden");
  }

  if(buyBtn){
    buyBtn.addEventListener("click", openModal);
  }

  if(closeBtn) closeBtn.addEventListener("click", closeModal);
  if(cancelBtn) cancelBtn.addEventListener("click", closeModal);

  if(confirmBtn){
    confirmBtn.addEventListener("click", () => {
      const symbol = $("tradeSymbol")?.innerText || (select ? select.value : "SFBT");
      const priceText = $("tradePrice")?.innerText || "0";
      const price = parseFloat(priceText);
      const qty = parseInt($("tradeQty")?.value, 10);

      const res = buy(symbol, price, qty);

      const msgEl = $("tradeMsg");
      if(msgEl){
        msgEl.innerText = res.msg;
        msgEl.style.color = res.ok ? "var(--good)" : "var(--bad)";
      }

      if(res.ok){
        // update funds display immediately
        const p = getPortfolio();
        if($("tradeFunds")) $("tradeFunds").innerText = money(p.funds);

        // close after a short delay
        setTimeout(closeModal, 900);
      }
    });
  }

  /* ===============================
     INIT
  =============================== */

  window.renderStockDemo = renderStock;
  renderStock();

});
