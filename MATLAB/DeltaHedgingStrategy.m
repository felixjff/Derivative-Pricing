%Purpose: Simulate a delta hedging strategy for a short position in a
%European Call option on a non-dividend paying stock. 
%Assumptions: 
%   1. The stock price follows a geometric brownian motion (GBM)
%   2. The option's delta sensitivity is based on the Black-Scholes Model

%% Setting 1: Parameters
r = 0.06;
vol = 0.2; %Volatility of stock price process matches the volatility used in the option valuation.
TDays = 252; %Maturity of one year, with 251 trading days
k = 99;
s_0 = 100;
cs = 100; %Option contract size

%% Stock paths simulation with dS_t = S_{t-1}(r dt + sigma dZ)
paths = 200;
s_t = zeros(TDays, paths);

%Initialization
s_t(1,:) = s_0;

for i = 1:paths
    for j = 2:TDays
       s_t(j,i) = s_t(j-1,i)*exp((r - (vol^2)/2) * (1/TDays) + ...
                                    (vol/sqrt(TDays))*normrnd(0,1)); 
    end
end

%%
plot(s_t);
title('Stock Price Simulation (GBM)');
xlabel('Trading Day');
ylabel('Stock Price');




%% Delta hedging strategy with daily hedge adjustment
ha_freq = TDays;
%Compute the hedge parameter implied by the Black-Scholes model
HP_t = zeros(TDays, paths);

for i = 1:paths
    for j = 1:TDays
       HP_t(j,i) = BS_HP_EUCall(s_t(j,i),k,r,vol,T); 
    end
end

%Compute option value at every point in time (Black-Scholes)
f_t = zeros(TDays, paths);

for i = 1:paths
    for j = 1:TDays
       f_t(j,i) = Black_Scholes_EUCall(s_t(j,i),k,r,vol,T); 
    end
end

%%
%Initialize delta hedging strategy at t=0
    %Cost is incurred every day due to daily rebalancing.
TC_t = zeros(TDays, paths);
    %To finance rebalancing, trader must borrow an amount B_t
B_t = zeros(TDays, paths);
    %Number of shares purchased chages according to hedge parameter at t
NS_t = zeros(TDays, paths);
    %Unless the portfolio is perfectly hedge (zero value at t) the
    %portfolio might have non-zero value at t. This is only observed one
    %period after rebalancing.
P_v_t = zeros(TDays-1, paths);
    %There is a cost of replication at time of hedging adjustment
TCR_t = zeros(TDays-1, paths);

    %Initialize variables
TC_t(1,:) = cs.*HP_t(1,:).*s_t(1,:);
B_t(1,:) = cs.*HP_t(1,:).*s_t(1,:) - cs.*f_t(1,:);
NS_t = cs.*HP_t(1,:);

for t = 2:TDays
   P_v_t(t,:) = -cs.*f_t(t,:) + cs.*HP_t(t-1,:).*s_t(t,:) - B_t(t-1,:).*exp(r/364);
   TCR_t(t,:) = cs.*(HP_t(t,:) - HP_t(t-1,:)).*s_t(t,:);
   NS_t(t,:) = cs.*HP_t(t,:);
   TC_t(t,:) = TC_t(t-1,:).*exp(r/364) + TCR_t(t,:);
   B_t(t,:) = cs.*HP_t(t,:).*s_t(t,:) - cs.*f_t(t,:);
end

%Average cost of strategy
mu_cost = mean(TC_t(TDays,:));
%Std. Dev. of strategy cost
sig_cost = std(TC_t(TDays,:));

%%
%Net gain of investment strategy of short European Call option plus daily
%delta hedging adjustment
%1. Net cost of strategy after exercising of option (note that the proceeds
    %acquired through DH per stock are always equal or lower to the strike price. 
    %proceed depends on value of stock at expiration)
DH_holdings = zeros(1,paths);
for i = 1:paths
    if s_t(TDays,i) > k
        DH_holdings(1,i) = k;
    else
        DH_holdings(1,i) = s_t(TDays,i);
    end
end
DH_holdings = DH_holdings .* cs;
    %Compute the net replication cost
 RC_T = TC_t(TDays,:) - DH_holdings(1,:);

%% 
%2. Compute profit of strategy by adding the compounded face value of the option
%premiums to the net cost.
NG = zeros(1,paths);
NG = f_t(1,:)*cs*exp(r) - RC_T;

%Average cost of strategy
mu_NG = mean(NG);
%Std. Dev. of strategy cost
sig_NG = std(NG);




%% Delta hedging strategy with weekly hedge adjustment

%Compute the hedge parameter implied by the Black-Scholes model
HP_t_w = zeros(TDays/7, paths);
for i = 1:paths
    m = 1;
    for j = 1:7:TDays
       HP_t_w(m,i) = BS_HP_EUCall(s_t(j,i),k,r,vol,T); 
       m = m+1;
    end
end

%Compute option value at every point in time (Black-Scholes)
f_t_w = zeros(TDays/7, paths);
s_t_w = zeros(TDays/7, paths);

for i = 1:paths
    m = 1;
    for j = 1:7:TDays
       f_t_w(m,i) = f_t(j,i); 
       s_t_w(m,i) = s_t(j,i);
       m = m+1;
    end
end

%%
%Initialize delta hedging strategy at t=0
    %Cost is incurred every day due to daily rebalancing.
TC_t_w = zeros(TDays/7, paths);
    %To finance rebalancing, trader must borrow an amount B_t
B_t_w = zeros(TDays/7, paths);
    %Number of shares purchased chages according to hedge parameter at t
NS_t_w = zeros(TDays/7, paths);
    %Unless the portfolio is perfectly hedge (zero value at t) the
    %portfolio might have non-zero value at t. This is only observed one
    %period after rebalancing.
P_v_t_w = zeros(TDays/7-1, paths);
    %There is a cost of replication at time of hedging adjustment
TCR_t_w = zeros(TDays/7-1, paths);

    %Initialize variables
TC_t_w(1,:) = cs.*HP_t_w(1,:).*s_t_w(1,:);
B_t_w(1,:) = cs.*HP_t_w(1,:).*s_t_w(1,:) - cs.*f_t_w(1,:);
NS_t_w = cs.*HP_t_w(1,:);

for t = 2:TDays/7
   P_v_t_w(t,:) = -cs.*f_t_w(t,:) + cs.*HP_t_w(t-1,:).*s_t_w(t,:) - B_t_w(t-1,:).*exp(r/364);
   TCR_t_w(t,:) = cs.*(HP_t_w(t,:) - HP_t_w(t-1,:)).*s_t_w(t,:);
   NS_t_w(t,:) = cs.*HP_t_w(t,:);
   TC_t_w(t,:) = TC_t_w(t-1,:).*exp(r/364) + TCR_t_w(t,:);
   B_t_w(t,:) = cs.*HP_t_w(t,:).*s_t_w(t,:) - cs.*f_t_w(t,:);
end

%Average cost of strategy
mu_cost_w = mean(TC_t_w(TDays/7,:));
%Std. Dev. of strategy cost
sig_cost_w = std(TC_t_w(TDays/7,:));

%%  Net gain of investment strategy of short European Call option plus daily
%delta hedging adjustment
DH_holdings_w = zeros(1,paths);
for i = 1:paths
    if s_t_w(TDays/7,i) > k
        DH_holdings_w(1,i) = k;
    else
        DH_holdings_w(1,i) = s_t_w(TDays/7,i);
    end
end
DH_holdings_w = DH_holdings_w .* cs;
    %Compute the net replication cost
 RC_T_w = TC_t_w(TDays/7,:) - DH_holdings_w(1,:);

%% 
%2. Compute profit of strategy by adding the compounded face value of the option
%premiums to the net cost.
NG_w = zeros(1,paths);
NG_w = f_t_w(1,:)*cs*exp(r) - RC_T_w;

%Average cost of strategy
mu_NG_w = mean(NG_w);
%Std. Dev. of strategy cost
sig_NG_w = std(NG_w);





%% Setting 2: Parameters
r = 0.06;
vol_s = 0:0.02:1; %Volatility of stock price process does not match the volatility used in the option valuation.
vol_o = 0.2;
TDays = 252; %Maturity of one year, with 251 trading days
k = 99;
s_0 = 100;
cs = 100; %Option contract size


