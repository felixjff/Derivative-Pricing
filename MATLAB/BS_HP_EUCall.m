%Purpose: Compute the hedge parameter for a portfolio of stock and European
%call using Black-Scholes formula.
function [output] = BS_HP_EUCall(s_0,k,r,vol,T)
  %Compute inputs for cumulative normal cdf
  d_ = (log(s_0/k) + (r - (vol.^2)./2).*T)./(vol*sqrt(T));
  d_1_ = d_ + vol*sqrt(T);
  %Compute price(s)
  hedge_parameter = normcdf(d_1_);
output = hedge_parameter;