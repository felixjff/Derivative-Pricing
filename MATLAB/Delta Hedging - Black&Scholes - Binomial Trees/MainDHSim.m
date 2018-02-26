%Purpose: Simulate a delta hedging strategy for a short position in a
%European Call option on a non-dividend paying stock. 
%Assumptions: 
%   1. The stock price follows a geometric brownian motion (GBM)
%   2. The option's delta sensitivity is based on the Black-Scholes Model

%% Setting 1: Parameters (Daily adjustment)
r = 0.06;
vol_o = 0.2; %Volatility of stock price process matches the volatility used in the option valuation.
vol_s = 0.2;
T = 1; %Maturity of one year, with 251 trading days
k = 99;
s_0 = 100;
hf = 1; %Daily hedging adjustment
paths = 1000; %Stock price paths

NG = zeros(2,1);
TC = zeros(2,1);
[s_t, f_t, HP_t, B_t, P_v_t, NG, TC] = DeltaHedgingSimulator(s_0, k, r, vol_s, vol_o, T, paths, hf);

%%
plot(s_t);
title('Stock Price Simulation (GBM)');
xlabel('Trading Day');
ylabel('Stock Price');

%% Dynamic Delta Hedging Ploted
prt = zeros(252,1);
for i = 1:251
prt(i) = f_t(i+1,1);
end
plot(1:251, prt(1:251), 1:251, HP_t(1:251,1).*s_t(1:251,1)-B_t(1:251,1).*exp(r/364))
title('Delta Hedging Strategy (Daily Hedge Adjustments)');
xlabel('Trading Day');
ylabel('Portfolio and Option Values');


%% Setting 2: Parameters (Weekly adjustment)
hf = 7; %Weekly hedging adjustments

NG_w = zeros(2,1);
TC_w = zeros(2,1);
[s_t_w, f_t_w, HP_t_w, B_t_w, P_v_t_w, NG_w, TC_w] = DeltaHedgingSimulator(s_0, k, r, vol_s, vol_o, T, paths, hf);


%% Dynamic Delta Hedging Ploted
prt = zeros(36,1);
prt(1) = f_t(1,1);
m =1;
for i = 1:7:252
prt(m) = f_t(i+1,1);
m=m+1;
end
plot(1:35, prt(1:35), 1:35, HP_t_w(1:35,1).*s_t_w(1:35,1)-B_t_w(1:35,1).*exp(r/364))
title('Delta Hedging Strategy');
xlabel('Trading Day');
ylabel('Portfolio and Option Values');

%% Comparison for different heding adjustment frequencies
%histogram(P_v_t(252,:),10);
%histogram(P_v_t_w(36,:),50);

%% Setting 2: Parameters (Daily adjustment + mismatching volatilities)
hf = 1; 
vol_o = 0.2;
vol_s = 0.2;

NG_vol = zeros(2,1);
TC_vol = zeros(2,1);
[s_t_vol, f_t_vol, HP_t_vol, B_t_vol, P_v_t_vol, NG_vol, TC_vol] = DeltaHedgingSimulator(s_0, k, r, vol_s, vol_o, T, paths, hf);


%% Note that the opton price derived above is not the actual option. This is
%because according to Black-Scholes formula, the price of the option should
%be derived using the actual volatility of the underlying and not an
%expectation. The actual option price is:
f_t_ac = zeros(252, paths);
for i = 1:paths
    for j = 1:252
      if j ==1 
       f_t_ac(j,i) = Black_Scholes_EUCall(s_t_vol(j,i),k,r,vol_s,T); 
      else
       f_t_ac(j,i) = Black_Scholes_EUCall(s_t_vol(j,i),k,r,vol_s,(252-j)/252);
      end
    end
end


%% Plot
prt = zeros(251,1);
prt(1) = f_t_vol(1,1);
m =1;
for i = 1:251
prt(m) = f_t_vol(i+1,1);
m=m+1;
end

plot(1:251, prt(1:251), 1:251, HP_t_vol(1:251,1).*s_t_vol(1:251,1)-B_t_vol(1:251,1).*exp(r/364), 1:251, f_t_ac(1:251,1))
hold on
legend('Expected Value', 'Portfolio Value', 'Actual Value')
hold on
title('Delta Hedging Strategy');
hold on
xlabel('Trading Day');
hold on
ylabel('Portfolio and Option Values');

%% Compute actual tracking error
%P_v_t_a = P_v_t_vol(252,:) + f_t(252,:) - f_t_ac(252,:); %Mismatching
P_v_t_a = P_v_t_vol(252,:); %Matching
%Average and std. from 1000 simulations
mean(P_v_t_a)
std(P_v_t_a)



%% Setting 2: Parameters (Daily adjustment + mismatching volatilities)
hf = 1; 
vol_o = 0.7;
vo_s = 0.2;

NG_vol1 = zeros(2,1);
TC_vol1 = zeros(2,1);
[s_t_vol1, f_t_vol1, HP_t_vol1, B_t_vol1, P_v_t_vol1, NG_vol1, TC_vol1] = DeltaHedgingSimulator(s_0, k, r, vol_s, vol_o, T, paths, hf);

%% Plot
prt = zeros(251,1);
prt(1) = f_t_vol(1,1);
m =1;
for i = 1:251
prt(m) = f_t_vol1(i+1,1);
m=m+1;
end

plot(1:251, prt(1:251), 1:251, HP_t_vol1(1:251,1).*s_t_vol1(1:251,1)-B_t_vol1(1:251,1).*exp(r/364))
title('Delta Hedging Strategy');
xlabel('Trading Day');
ylabel('Portfolio and Option Values');


