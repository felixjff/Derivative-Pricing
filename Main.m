%Purpose: Investigate the implied European Call price behavior for changing stock
%volatility.

%%Input Parameters
r = 0.06;
vol = 0.2;
T = 1;
N = 50;
k = 99;
s_0 = 100;

%% Vol = 20%

%Black Scholes
BS_f_v = Black_Scholes_EUCall(s_0,k,r,vol,T);
disp(sprintf('Value of European Call usng BS is %d', BS_f_v));

%Binomial Tree
BTM_f_v = BTM_EUCall(s_0,k,r,vol,T,N);
disp(sprintf('Value of European Call usng BTM is %d', BTM_f_v));

%% Volatility Grid
vol_ = 0:0.02:1;

%Analytical value (Black-Scholes)
BS_f_v_ = Black_Scholes_EUCall(s_0,k,r,vol_,T);

%Plot behavior of price as a function of stock volatility
plot(vol_,BS_f_v_);
title('Option Price (Black & Scholes)');
xlabel('Stock Volatility');
ylabel('Implied Option Value');

%%

%Value (Binomial Tree Model)
BTM_f_v_ = zeros(length(vol_),1)';
j = 1;
for i = vol_
    BTM_f_v_(j) = BTM_EUCall(s_0,k,r,i,T,N);
    j = j + 1;
end

%Plot behavior of price as a function of stock volatility
plot(vol_,BTM_f_v_);
title('Option Price (Binomial Tree Model)');
xlabel('Stock Volatility');
ylabel('Implied Option Value');

%% Comparison of values
diff_val = BTM_f_v_ - BS_f_v_;

plot(vol_,diff_val);
title('Binomial Tree vs Black-Scholes');
xlabel('Stock Volatility');
ylabel('Valuation Difference');

%% Convergence of Binomial Tree Model

%Steps grid
N_ = 1:100;

%Option Prices for different amount of steps
BTM_f_v_N = zeros(length(N_),1)';
j = 1;
for i = N_
    BTM_f_v_N(j) = BTM_EUCall(s_0,k,r,vol,T,i);
    j = j + 1;
end

plot(N_,BTM_f_v_N);
title('Convergence Analysis: Binomial Tree Model');
xlabel('Number of Steps');
ylabel('Implied Option Value');


%% Hedge Parameter Comparison

%Black-Scholes Model
BS_HP = BS_HP_EUCall(s_0,k,r,vol,T);

%Binomial-Tree Model
BTM_HP = BTM_HP_EUCall(s_0,k,r,vol,T, N);

%% Volatility Grid 

%Hedging Parameter (Black-Scholes)
BS_HP_ = BS_HP_EUCall(s_0,k,r,vol_,T);

%Plot behavior of price as a function of stock volatility
plot(vol_,BS_HP_);
title('Hedge Parameter (Black-Scholes)');
xlabel('Stock Volatility');
ylabel('Implied Hedge');


%% Hedging Parameter (Binomial Tree Model)
BTM_HP_ = zeros(length(vol_),1)';
j = 1;
for i = vol_
    BTM_HP_(j) = BTM_HP_EUCall(s_0,k,r,i,T,N);
    j = j + 1;
end

%Plot behavior of price as a function of stock volatility
plot(vol_,BTM_HP_);
title('Hedge Parameter (Binomial Tree Model)');
xlabel('Stock Volatility');
ylabel('Implied Hedge');    

%% Comparison of Hedge Parameter Values
diff_val_HP = BTM_HP_ - BS_HP_;

plot(vol_,diff_val_HP);
title('Binomial Tree vs Black-Scholes');
xlabel('Stock Volatility');
ylabel('Hedge Differences');





%% AMERICAN OPTIONS %%




%% American Call Option

%Binomial Tree with Stock Vol = 20%
BTM_f_v_A = BTM_AMCall(s_0,k,r,vol,T,N);
disp(sprintf('Value of American Call usng BTM is %d', BTM_f_v_A));

%Volatility Grid
BTM_f_v_A_ = zeros(length(vol_),1)';
j = 1;
for i = vol_
    BTM_f_v_A_(j) = BTM_AMCall(s_0,k,r,i,T,N);
    j = j + 1;
end

%% Plot behavior of price as a function of stock volatility
plot(vol_,BTM_f_v_A_);
title('American Call Option Price (Binomial Tree Model)');
xlabel('Stock Volatility');
ylabel('Implied Option Value');


%% American Put Option

%Binomial Tree with Stock Vol = 20%
BTM_f_v_AP = BTM_AMPut(s_0,k,r,vol,T,N);
disp(sprintf('Value of American Call usng BTM is %d', BTM_f_v_AP));

%Volatility Grid
BTM_f_v_AP_ = zeros(length(vol_),1)';
j = 1;
for i = vol_
    BTM_f_v_AP_(j) = BTM_AMPut(s_0,k,r,i,T,N);
    j = j + 1;
end

%% Plot behavior of price as a function of stock volatility
plot(vol_,BTM_f_v_AP_);
title('American Put Option Price (Binomial Tree Model)');
xlabel('Stock Volatility');
ylabel('Implied Option Value');




