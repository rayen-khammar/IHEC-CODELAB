/* ========= RISK FORMULA ========= */

function calculateReturns(prices){
  const r=[];
  for(let i=1;i<prices.length;i++){
    r.push((prices[i]-prices[i-1])/prices[i-1]);
  }
  return r;
}

function calculateVariance(arr){
  const mean=arr.reduce((a,b)=>a+b,0)/arr.length;
  return arr.reduce((s,v)=>s+(v-mean)*(v-mean),0)/arr.length;
}

function riskCoefficient(prices,userAgg=0.5){
  const returns=calculateReturns(prices);
  const variance=calculateVariance(returns);
  return variance*(1-userAgg);
}

/* ========= USER RISK ========= */

function getUserAgg(){
  const p=JSON.parse(localStorage.getItem("bvmt_profile_v2")||"{}");
  const score=p?.risk?.score||50;
  return score/100;
}

/* ========= DEMO PORTFOLIO ========= */

const portfolio=[
{
  name:"SFBT",
  price:12.4,
  quantity:120,
  history:[10,10.5,11,11.3,11.9,12.4]
},
{
  name:"BIAT",
  price:98,
  quantity:20,
  history:[90,92,94,95,97,98]
},
{
  name:"DELICE",
  price:9.2,
  quantity:200,
  history:[8.5,8.7,8.8,9,9.1,9.2]
}
];

function riskClass(v){
  if(v<0.0005)return"risk-low";
  if(v<0.0015)return"risk-mid";
  return"risk-high";
}

/* ========= RENDER ========= */

function renderPortfolio(){

  const grid=document.getElementById("portfolioGrid");

  const userAgg=getUserAgg();

  let total=0;
  grid.innerHTML="";

  portfolio.forEach(s=>{

    const risk=riskCoefficient(s.history,userAgg);
    const riskUI=risk*10000;

    total+=s.price*s.quantity;

    const card=document.createElement("div");
    card.className="stock-card";

    card.innerHTML=`
      <div class="stock-name">${s.name}</div>

      <div class="stock-price">
        ${s.price.toFixed(2)} TND
      </div>

      <div class="stock-meta">
        Qty ${s.quantity} • ${(s.price*s.quantity).toFixed(0)} TND
      </div>

      <div class="risk-box">
        Adjusted Risk:
        <b class="${riskClass(risk)}">
          ${riskUI.toFixed(2)}
        </b>
      </div>
    `;

    grid.appendChild(card);

  });

  document.getElementById("portfolioValue").innerText=
    total.toFixed(0)+" TND";

  const p=JSON.parse(localStorage.getItem("bvmt_profile_v2")||"{}");

  document.getElementById("userRiskScore").innerText=
    p?.risk?.score||"--";

  document.getElementById("userRiskLabel").innerText=
    p?.risk?.label||"--";

}

renderPortfolio();

