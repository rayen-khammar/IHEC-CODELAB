/* =================================
 GLOBAL PORTFOLIO STORE
 Used by ALL pages
================================= */

const PORTFOLIO_KEY = "BVMT_PORTFOLIO";

/* GET PORTFOLIO */
function getPortfolio(){

  let data = localStorage.getItem(PORTFOLIO_KEY);

  if(!data){

    const defaultPortfolio = {
      funds: 5288,   // matches your UI example
      holdings: []
    };

    localStorage.setItem(PORTFOLIO_KEY, JSON.stringify(defaultPortfolio));
    return defaultPortfolio;
  }

  return JSON.parse(data);
}

/* SAVE */
function savePortfolio(p){
  localStorage.setItem(PORTFOLIO_KEY, JSON.stringify(p));
}

/* BUY LOGIC */
function buyStock(symbol, price, qty){

  const p = getPortfolio();

  const totalCost = price * qty;

  if(p.funds < totalCost){
    return { success:false, message:"Not enough funds" };
  }

  p.funds -= totalCost;

  p.holdings.push({
    symbol,
    qty,
    price
  });

  savePortfolio(p);

  return { success:true, newFunds:p.funds };
}

window.portfolioStore = {
  getPortfolio,
  buyStock,
  savePortfolio
};
