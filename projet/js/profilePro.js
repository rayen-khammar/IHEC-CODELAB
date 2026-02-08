const STORAGE = "user_fin_profile";

/* ===========================
   FAVORITE STOCKS
=========================== */

function addFavorite() {
    const val = document.getElementById("favInput").value.toUpperCase();
    if(!val) return;

    const data = loadProfile();
    data.favorites = data.favorites || [];

    if(!data.favorites.includes(val))
        data.favorites.push(val);

    saveProfile(data);
    renderFavorites();
}

function renderFavorites(){
    const data = loadProfile();
    const list = document.getElementById("favList");

    list.innerHTML = "";

    (data.favorites || []).forEach(stock=>{
        const div = document.createElement("div");
        div.innerHTML = `${stock} <button onclick="removeFav('${stock}')">X</button>`;
        list.appendChild(div);
    });
}

function removeFav(sym){
    const data = loadProfile();
    data.favorites = data.favorites.filter(x=>x!==sym);
    saveProfile(data);
    renderFavorites();
}

/* ===========================
   STORAGE
=========================== */

function loadProfile(){
    return JSON.parse(localStorage.getItem(STORAGE) || "{}");
}

function saveProfile(data){
    localStorage.setItem(STORAGE, JSON.stringify(data));
}

/* ===========================
   RISK FORMULA
=========================== */

function calculateRisk(){

    const age = Number(val("age"));
    const budget = Number(val("budget"));
    const horizon = Number(val("horizon"));
    const experience = Number(val("experience"));

    const lossReaction = val("lossReaction");
    const incomeStability = val("incomeStability");
    const liquidityNeed = val("liquidityNeed");

    const stablePct = Number(val("stablePct"));
    const bondPct = Number(val("bondPct"));
    const cashPct = Number(val("cashPct"));
    const riskPct = Number(val("riskPct"));

    let score = 0;

    /* AGE */
    if(age < 30) score += 10;
    else if(age < 45) score += 7;
    else score += 4;

    /* BUDGET */
    if(budget > 50000) score += 15;
    else if(budget > 10000) score += 10;
    else score += 5;

    /* HORIZON */
    score += horizon;

    /* EXPERIENCE */
    score += experience * 4;

    /* LOSS REACTION */
    if(lossReaction === "buy") score += 10;
    if(lossReaction === "hold") score += 6;
    if(lossReaction === "sell") score += 2;

    /* INCOME STABILITY */
    if(incomeStability === "stable") score += 10;
    if(incomeStability === "medium") score += 6;
    if(incomeStability === "unstable") score += 3;

    /* LIQUIDITY NEED */
    if(liquidityNeed === "low") score += 10;
    if(liquidityNeed === "medium") score += 6;
    if(liquidityNeed === "high") score += 2;

    /* PORTFOLIO RISK */
    score += (riskPct * 0.2);

    score = Math.min(100, score);

    document.getElementById("riskScore").innerText = score;

    let label = "Moderate";

    if(score < 35) label = "Low Risk";
    else if(score < 65) label = "Moderate Risk";
    else label = "High Risk";

    document.getElementById("riskLabel").innerText = label;

    saveAllInputs(score,label);
}

/* ===========================
   SAVE INPUTS
=========================== */

function saveAllInputs(score,label){
    const data = loadProfile();

    data.inputs = {
        name: val("name"),
        age: val("age"),
        income: val("income"),
        budget: val("budget"),
        horizon: val("horizon"),
        experience: val("experience"),
        lossReaction: val("lossReaction"),
        incomeStability: val("incomeStability"),
        liquidityNeed: val("liquidityNeed"),
        stablePct: val("stablePct"),
        bondPct: val("bondPct"),
        cashPct: val("cashPct"),
        riskPct: val("riskPct")
    };

    data.risk = { score, label };

    saveProfile(data);
}

/* ===========================
   HELPERS
=========================== */

function val(id){
    return document.getElementById(id).value;
}

/* ===========================
   INIT
=========================== */

window.onload = ()=>{
    renderFavorites();
};
