export function calculateReturns(prices) {
  const returns = [];
  for (let i = 1; i < prices.length; i++) {
    returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
  }
  return returns;
}

export function calculateVariance(arr) {
  const mean = arr.reduce((sum, val) => sum + val, 0) / arr.length;
  const variance = arr.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / arr.length;
  return variance;
}

export function riskCoefficient(prices, userAggressiveness = 0.5) {
  const returns = calculateReturns(prices);
  const variance = calculateVariance(returns);
  return variance * (1 - userAggressiveness);
}
