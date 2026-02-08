const LS_KEY = "bvmt_profile_v2";

/* ---------------------------
   Helpers
--------------------------- */
const $ = (id) => document.getElementById(id);
const errBox = (forId) => document.querySelector(`[data-error-for="${forId}"]`);

function clamp(n, a, b){ return Math.max(a, Math.min(b, n)); }
function toInt(v){ const n = Number(v); return Number.isFinite(n) ? Math.trunc(n) : NaN; }
function toNum(v){ const n = Number(v); return Number.isFinite(n) ? n : NaN; }

function setError(id, msg){
  const box = errBox(id);
  if (box) box.textContent = msg || "";
  const el = $(id);
  if (el) el.classList.toggle("is-invalid", Boolean(msg));
}

function toast(message, tone="info"){
  const host = document.getElementById("toast-host");
  if (!host) return;
  const el = document.createElement("div");
  el.className = `toast toast--${tone}`;
  el.textContent = message;
  host.appendChild(el);
  setTimeout(() => el.classList.add("toast--show"), 10);
  setTimeout(() => {
    el.classList.remove("toast--show");
    setTimeout(() => el.remove(), 220);
  }, 2200);
}

/* ---------------------------
   Favorites (chips UX)
--------------------------- */
function normalizeTicker(s){
  return (s || "").trim().toUpperCase().replace(/[^A-Z0-9.]/g, "");
}

function renderFavs(favs){
  const host = $("favChips");
  host.innerHTML = "";

  if (!favs.length){
    host.innerHTML = `<div class="p-hint">No favorites yet. Add tickers like SFBT, BIAT…</div>`;
    return;
  }

  favs.forEach(sym => {
    const chip = document.createElement("div");
    chip.className = "chip-tag";
    chip.innerHTML = `<span>${sym}</span><button type="button" aria-label="Remove ${sym}">×</button>`;
    chip.querySelector("button").addEventListener("click", () => {
      const profile = readProfile();
      profile.favorites = (profile.favorites || []).filter(x => x !== sym);
      writeProfile(profile);
      renderFavs(profile.favorites);
      computeAndRender(); // update suggestions if needed
    });
    host.appendChild(chip);
  });
}

function addFav(){
  setError("favorites", "");
  const raw = $("favInput").value;
  const sym = normalizeTicker(raw);
  if (!sym) return setError("favorites", "Enter a ticker (letters/numbers only).");

  const profile = readProfile();
  profile.favorites = profile.favorites || [];

  if (profile.favorites.includes(sym)){
    $("favInput").value = "";
    return toast("Already added.", "info");
  }

  if (profile.favorites.length >= 12){
    return setError("favorites", "Max 12 favorites for clarity.");
  }

  profile.favorites.push(sym);
  writeProfile(profile);
  $("favInput").value = "";
  renderFavs(profile.favorites);
  toast("Favorite added.", "ok");
}

/* ---------------------------
   Allocation (must total 100)
   Auto-balance remainder into Cash
--------------------------- */
const allocKeys = ["Stable", "Bonds", "Cash", "Risk"];

function getAlloc(){
  return {
    Stable: clamp(toInt($("allocStableNum").value), 0, 100),
    Bonds: clamp(toInt($("allocBondsNum").value), 0, 100),
    Cash: clamp(toInt($("allocCashNum").value), 0, 100),
    Risk: clamp(toInt($("allocRiskNum").value), 0, 100),
  };
}

function setAlloc(a){
  $("allocStable").value = a.Stable; $("allocStableNum").value = a.Stable;
  $("allocBonds").value = a.Bonds;   $("allocBondsNum").value = a.Bonds;
  $("allocCash").value = a.Cash;     $("allocCashNum").value = a.Cash;
  $("allocRisk").value = a.Risk;     $("allocRiskNum").value = a.Risk;
  updateAllocTotal();
}

function updateAllocTotal(){
  const a = getAlloc();
  const total = (a.Stable||0) + (a.Bonds||0) + (a.Cash||0) + (a.Risk||0);
  $("allocTotal").textContent = String(total);
  setError("allocation", total === 100 ? "" : "Allocation must sum to 100%.");
  return total;
}

// When user changes any allocation, we auto-fix by adjusting Cash to keep total 100.
function autoBalance(changedKey){
  const a = getAlloc();

  // Convert NaN to 0
  for (const k of allocKeys) if (!Number.isFinite(a[k])) a[k] = 0;

  // Clamp
  for (const k of allocKeys) a[k] = clamp(a[k], 0, 100);

  // Sum others then set Cash = 100 - others (but not below 0)
  const others = (changedKey === "Cash")
    ? (a.Stable + a.Bonds + a.Risk)
    : (a.Stable + a.Bonds + a.Risk);

  let cash = 100 - others;
  cash = clamp(cash, 0, 100);
  a.Cash = cash;

  // If Cash got clamped to 0 but total > 100, reduce Risk first, then Stable, then Bonds
  let total = a.Stable + a.Bonds + a.Cash + a.Risk;
  if (total > 100){
    let overflow = total - 100;
    const order = ["Risk", "Stable", "Bonds"];
    for (const k of order){
      const reducible = Math.min(a[k], overflow);
      a[k] -= reducible;
      overflow -= reducible;
      if (overflow <= 0) break;
    }
  }

  setAlloc(a);
  saveAllocToProfile(a);
}

function saveAllocToProfile(a){
  const p = readProfile();
  p.allocation = a;
  writeProfile(p);
}

/* ---------------------------
   Risk scoring (0–100)
   Weighted + explainable
--------------------------- */
function computeRiskScore(profile){
  // Validation required fields:
  const age = toInt(profile.age);
  const inc = toNum(profile.monthlyIncome);
  const bud = toNum(profile.investmentBudget);

  // Normalize categorical inputs (0..2 or 0..3)
  const horizon = clamp(toInt(profile.horizon), 0, 3);        // 0..3
  const exp = clamp(toInt(profile.experience), 0, 2);         // 0..2
  const loss = clamp(toInt(profile.lossReaction), 0, 2);      // 0..2
  const liqNeed = clamp(toInt(profile.liquidityNeed), 0, 2);  // 0..2 (2 low need = more risk capacity)
  const incStab = clamp(toInt(profile.incomeStability), 0, 2);// 0..2
  const goal = clamp(toInt(profile.goal), 0, 2);              // 0..2

  const a = profile.allocation || { Stable:40, Bonds:30, Cash:30, Risk:0 };
  const allocRisk = clamp(toNum(a.Risk), 0, 100);
  const allocCash = clamp(toNum(a.Cash), 0, 100);

  // Age factor: younger -> higher capacity
  // 16..90 mapped to 1..0
  const ageNorm = Number.isFinite(age) ? (1 - ((clamp(age, 16, 90) - 16) / (90 - 16))) : 0;

  // Budget factor: log-ish scaling (avoid whales dominating)
  const budgetNorm = Number.isFinite(bud) ? clamp(Math.log10(bud + 1) / 7, 0, 1) : 0; // 0..~1

  // Income stability & liquidity: stable and low need -> higher capacity
  const incomeNorm = Number.isFinite(inc) ? clamp(Math.log10(inc + 1) / 6, 0, 1) : 0;

  // Horizon: longer horizon -> more capacity (0..3 -> 0..1)
  const horizonNorm = horizon / 3;

  // Experience: (0..2 -> 0..1)
  const expNorm = exp / 2;

  // Loss reaction: sell=0, hold=0.5, buy=1
  const lossNorm = loss / 2;

  // Liquidity need: (high=0 .. low=1)
  const liqNorm = liqNeed / 2;

  // Income stability: (unstable=0 .. stable=1)
  const stabNorm = incStab / 2;

  // Goal: preservation=0 .. aggressive=1
  const goalNorm = goal / 2;

  // Allocation realism:
  // - More high-risk allocation increases risk appetite score
  // - Too much cash reduces appetite
  const allocNorm = clamp((allocRisk / 100) * 0.9 + (1 - allocCash / 100) * 0.1, 0, 1);

  // Weighted score
  const parts = [
    { name: "Horizon", w: 0.15, v: horizonNorm },
    { name: "Age", w: 0.10, v: ageNorm },
    { name: "Budget", w: 0.10, v: budgetNorm },
    { name: "Income", w: 0.10, v: incomeNorm },
    { name: "Stability", w: 0.10, v: stabNorm },
    { name: "Liquidity", w: 0.10, v: liqNorm },
    { name: "Experience", w: 0.10, v: expNorm },
    { name: "Loss reaction", w: 0.10, v: lossNorm },
    { name: "Goal", w: 0.05, v: goalNorm },
    { name: "Allocation", w: 0.10, v: allocNorm },
  ];

  const score01 = parts.reduce((acc, p) => acc + p.w * p.v, 0);
  const score = Math.round(score01 * 100);

  let label = "Moderate";
  if (score < 35) label = "Conservative";
  else if (score < 70) label = "Moderate";
  else label = "Aggressive";

  return { score, label, parts };
}

function suggestedAllocation(label){
  // Simple, believable defaults for demo
  if (label === "Conservative"){
    return { Stable: 35, Bonds: 45, Cash: 15, Risk: 5 };
  }
  if (label === "Aggressive"){
    return { Stable: 40, Bonds: 10, Cash: 10, Risk: 40 };
  }
  return { Stable: 40, Bonds: 30, Cash: 20, Risk: 10 };
}

function renderRisk(r){
  $("riskScore").textContent = String(r.score);
  $("riskLabel").textContent = `${r.label} profile`;

  $("riskBar").style.width = `${clamp(r.score, 0, 100)}%`;

  const badge = $("riskBadge");
  badge.textContent = r.label;

  badge.style.borderColor = "rgba(255,255,255,0.10)";
  badge.style.background = "rgba(255,255,255,0.04)";
  badge.style.color = "var(--text)";

  if (r.label === "Conservative"){
    badge.style.borderColor = "rgba(34,197,94,0.25)";
    badge.style.background = "rgba(34,197,94,0.10)";
  } else if (r.label === "Aggressive"){
    badge.style.borderColor = "rgba(239,68,68,0.25)";
    badge.style.background = "rgba(239,68,68,0.10)";
  } else {
    badge.style.borderColor = "rgba(245,158,11,0.25)";
    badge.style.background = "rgba(245,158,11,0.10)";
  }

  // Explanation: show top 3 drivers
  const top = [...r.parts].sort((a,b)=> (b.w*b.v) - (a.w*a.v)).slice(0,3);
  $("riskExplain").innerHTML =
    `<strong>Top drivers:</strong><br>` +
    top.map(t => `• ${t.name}`).join("<br>") +
    `<br><br><span class="p-hint">Score updates as you edit.</span>`;

  // Suggested allocation
  const s = suggestedAllocation(r.label);
  $("suggestedAlloc").innerHTML = `
    <div class="suggest-item"><span>Stable stocks</span><strong>${s.Stable}%</strong></div>
    <div class="suggest-item"><span>Bonds</span><strong>${s.Bonds}%</strong></div>
    <div class="suggest-item"><span>Liquidity</span><strong>${s.Cash}%</strong></div>
    <div class="suggest-item"><span>High growth</span><strong>${s.Risk}%</strong></div>
  `;

  // Save computed risk
  const p = readProfile();
  p.risk = { score: r.score, label: r.label, parts: r.parts };
  writeProfile(p);
}

/* ---------------------------
   Validation (no negative values)
--------------------------- */
function readForm(){
  return {
    fullName: ($("fullName").value || "").trim(),
    age: $("age").value,
    monthlyIncome: $("monthlyIncome").value,
    investmentBudget: $("investmentBudget").value,
    horizon: $("horizon").value,
    experience: $("experience").value,
    lossReaction: $("lossReaction").value,
    liquidityNeed: $("liquidityNeed").value,
    incomeStability: $("incomeStability").value,
    goal: $("goal").value,
    allocation: getAlloc(),
    favorites: (readProfile().favorites || []),
  };
}

function validate(form){
  let ok = true;

  setError("fullName", "");
  setError("age", "");
  setError("monthlyIncome", "");
  setError("investmentBudget", "");
  setError("allocation", "");
  setError("favorites", "");

  if (form.fullName.length > 0 && form.fullName.length < 2){
    ok = false; setError("fullName", "Name is too short.");
  }

  const age = toInt(form.age);
  if (!Number.isFinite(age)) { ok = false; setError("age", "Enter a valid age."); }
  else if (age < 16 || age > 90) { ok = false; setError("age", "Age must be 16–90."); }

  const inc = toNum(form.monthlyIncome);
  if (!Number.isFinite(inc) || inc < 0){ ok = false; setError("monthlyIncome", "Income must be 0 or more."); }

  const bud = toNum(form.investmentBudget);
  if (!Number.isFinite(bud) || bud < 0){ ok = false; setError("investmentBudget", "Budget must be 0 or more."); }

  // Allocation must sum to 100
  const a = form.allocation;
  const total = (a.Stable||0) + (a.Bonds||0) + (a.Cash||0) + (a.Risk||0);
  $("allocTotal").textContent = String(total);
  if (total !== 100) { ok = false; setError("allocation", "Allocation must sum to 100%."); }

  return ok;
}

/* ---------------------------
   Storage
--------------------------- */
function readProfile(){
  try { return JSON.parse(localStorage.getItem(LS_KEY) || "{}"); }
  catch { return {}; }
}
function writeProfile(p){
  localStorage.setItem(LS_KEY, JSON.stringify(p));
}

/* ---------------------------
   Actions
--------------------------- */
function save(){
  const form = readForm();
  if (!validate(form)){
    toast("Please fix the highlighted fields.", "error");
    return false;
  }

  const p = readProfile();
  p.personal = {
    fullName: form.fullName,
    age: toInt(form.age),
    monthlyIncome: toNum(form.monthlyIncome),
    investmentBudget: toNum(form.investmentBudget),
  };
  p.drivers = {
    horizon: toInt(form.horizon),
    experience: toInt(form.experience),
    lossReaction: toInt(form.lossReaction),
    liquidityNeed: toInt(form.liquidityNeed),
    incomeStability: toInt(form.incomeStability),
    goal: toInt(form.goal),
  };
  p.allocation = form.allocation;

  writeProfile(p);
  toast("Profile saved.", "ok");
  return true;
}

function applyFromStorage(){
  const p = readProfile();

  // Personal
  $("fullName").value = p.personal?.fullName || "";
  $("age").value = p.personal?.age ?? "";
  $("monthlyIncome").value = p.personal?.monthlyIncome ?? "";
  $("investmentBudget").value = p.personal?.investmentBudget ?? "";

  // Drivers
  $("horizon").value = String(p.drivers?.horizon ?? 1);
  $("experience").value = String(p.drivers?.experience ?? 0);
  $("lossReaction").value = String(p.drivers?.lossReaction ?? 1);
  $("liquidityNeed").value = String(p.drivers?.liquidityNeed ?? 1);
  $("incomeStability").value = String(p.drivers?.incomeStability ?? 1);
  $("goal").value = String(p.drivers?.goal ?? 1);

  // Allocation
  const a = p.allocation || { Stable: 40, Bonds: 30, Cash: 20, Risk: 10 };
  setAlloc({
    Stable: clamp(toInt(a.Stable), 0, 100) || 0,
    Bonds: clamp(toInt(a.Bonds), 0, 100) || 0,
    Cash: clamp(toInt(a.Cash), 0, 100) || 0,
    Risk: clamp(toInt(a.Risk), 0, 100) || 0,
  });

  // Favorites
  p.favorites = p.favorites || [];
  writeProfile(p);
  renderFavs(p.favorites);

  // Risk
  if (p.risk?.score != null){
    renderRisk({ score: p.risk.score, label: p.risk.label, parts: p.risk.parts || [] });
  } else {
    computeAndRender();
  }
}

function computeAndRender(){
  const form = readForm();
  // Only compute if basic numeric fields are valid-ish (avoid NaN spam)
  const age = toInt(form.age);
  const inc = toNum(form.monthlyIncome);
  const bud = toNum(form.investmentBudget);

  // Still render allocation total live
  updateAllocTotal();

  if (!Number.isFinite(age) || age < 16 || age > 90) return;
  if (!Number.isFinite(inc) || inc < 0) return;
  if (!Number.isFinite(bud) || bud < 0) return;

  const profile = {
    age,
    monthlyIncome: inc,
    investmentBudget: bud,
    horizon: form.horizon,
    experience: form.experience,
    lossReaction: form.lossReaction,
    liquidityNeed: form.liquidityNeed,
    incomeStability: form.incomeStability,
    goal: form.goal,
    allocation: form.allocation,
  };

  const r = computeRiskScore(profile);
  renderRisk(r);
}

/* ---------------------------
   Events (live updates)
--------------------------- */
function bind(){
  // Save/Reset
  $("btn-save").addEventListener("click", () => { if (save()) computeAndRender(); });
  $("btnCompute").addEventListener("click", () => { if (save()) computeAndRender(); });
  $("btn-reset").addEventListener("click", () => {
    localStorage.removeItem(LS_KEY);
    toast("Profile reset.", "ok");
    location.reload();
  });

  // Favorites
  $("btnAddFav").addEventListener("click", addFav);
  $("favInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter"){
      e.preventDefault();
      addFav();
    }
  });

  // Allocation sliders + numbers
  const pairs = [
    ["allocStable", "allocStableNum", "Stable"],
    ["allocBonds", "allocBondsNum", "Bonds"],
    ["allocCash", "allocCashNum", "Cash"],
    ["allocRisk", "allocRiskNum", "Risk"],
  ];

  pairs.forEach(([rangeId, numId, key]) => {
    $(rangeId).addEventListener("input", () => {
      $(numId).value = $(rangeId).value;
      autoBalance(key);
      computeAndRender();
    });
    $(numId).addEventListener("input", () => {
      const v = clamp(toInt($(numId).value), 0, 100);
      $(rangeId).value = String(Number.isFinite(v) ? v : 0);
      autoBalance(key);
      computeAndRender();
    });
  });

  // Auto-fill
  $("btn-auto").addEventListener("click", () => {
    // Set a reasonable default and balance cash
    const p = readProfile();
    const base = p.risk?.label ? suggestedAllocation(p.risk.label) : { Stable:40, Bonds:30, Cash:20, Risk:10 };
    setAlloc(base);
    saveAllocToProfile(base);
    computeAndRender();
    toast("Auto-filled allocation.", "ok");
  });

  // Live compute on form changes
  const liveIds = ["fullName","age","monthlyIncome","investmentBudget","horizon","experience","lossReaction","liquidityNeed","incomeStability","goal"];
  liveIds.forEach(id => $(id).addEventListener("input", () => {
    // Clear negative immediately (extra safety)
    if (["age","monthlyIncome","investmentBudget"].includes(id)){
      const el = $(id);
      if (el.value && Number(el.value) < 0) el.value = "0";
    }
    computeAndRender();
  }));
}

/* ---------------------------
   Init
--------------------------- */
(function init(){
  // initial allocation defaults (will be overwritten by storage)
  setAlloc({ Stable: 40, Bonds: 30, Cash: 20, Risk: 10 });
  bind();
  applyFromStorage();
})();

