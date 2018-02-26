%Purpose: Simulate a delta hedging strategy over several stock paths and
%for different expected, actual volatility values. 

function [s_t, f_t, HP_t, B_t, P_v_t, Gains, Costs] = DeltaHedgingSimulator(s_0, k, r, vol_s, vol_o, T, paths, hf)
%Simulate stock price paths (252 trading days)
TDays = 252*T;
s_t = zeros(TDays, paths);
%Initialization
s_t(1,:) = s_0;
for i = 1:paths
    for j = 2:TDays
       s_t(j,i) = s_t(j-1,i)*exp((r - (vol_s^2)/2) * (1/TDays) + ...
                                    (vol_s/sqrt(TDays))*normrnd(0,1)); 
    end
end

%Compute option value at every point in time (Black-Scholes)
f_t = zeros(TDays, paths);

for i = 1:paths
    for j = 1:TDays
      if j ==1 
       f_t(j,i) = Black_Scholes_EUCall(s_t(j,i),k,r,vol_o,T); 
      else
       f_t(j,i) = Black_Scholes_EUCall(s_t(j,i),k,r,vol_o,(TDays-j)/TDays);
      end
    end
end

%Compute the hedge parameter implied by the Black-Scholes model and
%volatility expectations 
HP_t = zeros(TDays/hf, paths);
for i = 1:paths
    m = 1;
    for j = 1:hf:TDays
        if j==1
         HP_t(m,i) = BS_HP_EUCall(s_t(j,i),k,r,vol_o,T);
        else
         HP_t(m,i) = BS_HP_EUCall(s_t(j,i),k,r,vol_o,(TDays-j)/TDays); 
         m = m+1;
        end
    end
end

%If the volatility of the underlying is different than expected also
%compute true value of option.
if vol_s ~= vol_o
    f_t_a = zeros(TDays, paths);

  for i = 1:paths
    for j = 1:TDays
      if j ==1 
       f_t_a(j,i) = Black_Scholes_EUCall(s_t(j,i),k,r,vol_s,T); 
      else
       f_t_a(j,i) = Black_Scholes_EUCall(s_t(j,i),k,r,vol_s,(TDays-j)/TDays);
      end
    end
  end
end

%% Initialize delta hedging strategy at t=0
    %Cost is incurred every day due to daily rebalancing.
TC_t = zeros(TDays/hf, paths);
    %To finance rebalancing, trader must borrow an amount B_t
B_t = zeros(TDays/hf, paths);
    %Number of shares purchased chages according to hedge parameter at t
NS_t = zeros(TDays/hf, paths);
    %Unless the portfolio is perfectly hedge (zero value at t) the
    %portfolio might have non-zero value at t. This is only observed one
    %period after rebalancing. This value s also called "tracking error".
P_v_t = zeros(TDays/hf-1, paths);
    %There is a cost of replication at time of hedging adjustment
TCR_t = zeros(TDays/hf-1, paths);
    %Initialize variables
TC_t(1,:) = HP_t(1,:).*s_t(1,:);
B_t(1,:) = HP_t(1,:).*s_t(1,:) - f_t(1,:);
NS_t = HP_t(1,:);

    %Costs and values
j = hf;   
for t = 2:TDays/hf
   P_v_t(t,:) = HP_t(t-1,:).*s_t(j,:) - f_t(j,:) - B_t(t-1,:).*exp(r/364);
   TCR_t(t,:) = (HP_t(t,:) - HP_t(t-1,:)).*s_t(j,:);
   NS_t(t,:) = HP_t(t,:);
   TC_t(t,:) = TC_t(t-1,:).*exp(r/364) + TCR_t(t,:);
   B_t(t,:) = HP_t(t,:).*s_t(j,:) - f_t(j,:);
   j = j + hf;
end

%Average cost of strategy
mu_cost = mean(TC_t(TDays/hf,:));
%Std. Dev. of strategy cost
std_cost = std(TC_t(TDays/hf,:));

Costs = [mu_cost, std_cost];

%Net gain of investment strategy of short European Call option plus daily delta hedging adjustment
%1. Net cost of strategy after exercising of option (note that the proceeds acquired through DH per stock are always 
    %equal or lower to the strike price. Proceed depends on value of stock at expiration)
DH_holdings = zeros(1,paths);
for i = 1:paths
    if s_t(TDays,i) > k
        DH_holdings(1,i) = k;
    else
        DH_holdings(1,i) = s_t(TDays,i);
    end
end
    %Compute the net replication cost
RC_T = TC_t(TDays/hf,:) - DH_holdings(1,:);
%2. Compute profit of strategy by adding the compounded face value of the option premiums to the net cost.
NG = zeros(1,paths);
NG = f_t(1,:)*exp(r) - RC_T;
%Average cost of strategy
mu_NG = mean(NG);
%Std. Dev. of strategy cost
std_NG = std(NG);
Gains = [mu_NG, std_NG];

end