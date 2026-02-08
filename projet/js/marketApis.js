/*
==============================
 MARKET API LAYER
 All backend calls live here
==============================
*/

const API_BASE = "https://YOUR_BACKEND_URL";

/* =============================
   INDICES
============================= */

async function fetchIndices(){

  /*
  FUTURE REAL API:

  return fetch(API_BASE+"/market/indices")
    .then(r=>r.json());
  */

  // DEMO DATA
  return {
    tunindex:{ value: 9230, trend: "+0.8%" },
    tunindex20:{ value: 4021, trend: "-0.2%" }
  };
}

/* =============================
   SENTIMENT
============================= */

async function fetchSentiment(){
  return {
    label:"Bullish",
    score:72
  };
}

/* =============================
   MOVERS
============================= */

async function fetchMovers(){
  return {
    gainers:[
      {symbol:"SFBT", change:"+3.2%"},
      {symbol:"BIAT", change:"+2.1%"}
    ],
    losers:[
      {symbol:"TPR", change:"-2.4%"},
      {symbol:"ASSAD", change:"-1.9%"}
    ]
  };
}

/* =============================
   ALERTS
============================= */

async function fetchAlerts(){
  return [
    "Unusual volume detected on BIAT",
    "Price spike detected on DELICE"
  ];
}

window.marketApi = {
  fetchIndices,
  fetchSentiment,
  fetchMovers,
  fetchAlerts
};
