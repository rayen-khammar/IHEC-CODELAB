/* =========================
   LOAD DATA + RENDER UI
========================= */

async function loadDashboard(){

  const indicesRaw = await marketApi.fetchIndices();
  const sentimentRaw = await marketApi.fetchSentiment();
  const moversRaw = await marketApi.fetchMovers();
  const alertsRaw = await marketApi.fetchAlerts();

  const indices = dashboardMapper.mapIndices(indicesRaw);
  const sentiment = dashboardMapper.mapSentiment(sentimentRaw);
  const movers = dashboardMapper.mapMovers(moversRaw);
  const alerts = dashboardMapper.mapAlerts(alertsRaw);

  renderIndices(indices);
  renderSentiment(sentiment);
  renderMovers(movers);
  renderAlerts(alerts);
}

/* =========================
   RENDER FUNCTIONS
========================= */

function renderIndices(data){
  document.getElementById("tunindexVal").innerText = data.tunindex.value;
  document.getElementById("tunindexTrend").innerText = data.tunindex.trend;

  document.getElementById("tun20Val").innerText = data.tunindex20.value;
  document.getElementById("tun20Trend").innerText = data.tunindex20.trend;
}

function renderSentiment(data){
  document.getElementById("sentimentVal").innerText = data.label;
  document.getElementById("sentimentScore").innerText = "Score "+data.score;
}

function renderMovers(data){

  const gainers = document.getElementById("gainersTable");
  gainers.innerHTML="";

  data.gainers.forEach(g=>{
    gainers.innerHTML += `
      <tr>
        <td>${g.symbol}</td>
        <td>${g.change}</td>
      </tr>
    `;
  });

  const losers = document.getElementById("losersTable");
  losers.innerHTML="";

  data.losers.forEach(l=>{
    losers.innerHTML += `
      <tr>
        <td>${l.symbol}</td>
        <td>${l.change}</td>
      </tr>
    `;
  });

}

function renderAlerts(data){

  const container = document.getElementById("alertsList");
  container.innerHTML="";

  data.forEach(a=>{
    container.innerHTML += `<div class="mini">${a}</div>`;
  });

}

/* =========================
   EVENTS
========================= */

document.getElementById("refreshBtn").onclick = loadDashboard;

document.getElementById("quickOpenBtn").onclick = ()=>{
  const s = document.getElementById("quickSymbol").value;
  if(s) location.href = "./stock.html?symbol="+s;
};

/* INIT */
loadDashboard();

