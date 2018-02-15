%Purpose: Price option using Black-Scholes' analytical solution
function [output] = Black_Scholes_EUCall(s_0,k,r,vol,T)
  %Compute inputs for cumulative normal cdf
  d_ = (log(s_0/k) + (r - (vol.^2)./2).*T)./(vol*sqrt(T));
  d_1_ = d_ + vol*sqrt(T);
  %Compute price(s)
  BS_f_v = s_0.*normcdf(d_1_) - k*exp(-r*T).*normcdf(d_);
output = BS_f_v;