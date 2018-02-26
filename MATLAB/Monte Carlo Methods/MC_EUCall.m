%Purpose: Price a European put option using Monte Carlo methods and analyze
%the results for changing parameter values. Also, compare the results with
%the results obtained from using other pricing methods. 

%% Parameters
T = 1;
k = 99;
r = 0.06;
s_0 = 100;
vol = 0.2;

%% Put price (Analytical solution)
BS_f_v = Black_Scholes_EUPut(s_0,k,r,vol,T);


%% Monte Carlo Simulation of end price 
%(result follows from applying Euler method and integration)

%% Convergence analysis (Changing Number of Path Realizations, 1000 Trials)
sim = 5000:5000:50000;
s_T = NaN(sim(length(sim)), length(sim));
payoff = NaN(sim(length(sim)), length(sim));
for m = 1:length(sim)
        for j = 1:sim(m)
            %Compute stock price at T
            s_T(j,m) = s_0*exp((r-0.5*vol^2)*T + vol*sqrt(T)*normrnd(0,1));
            %Compute discounted payoff
            if(k - s_T(j,m) > 0)
                payoff(j,m) = exp(-r*T)*(k - s_T(j,m));
            else
                payoff(j,m) = 0;
            end
        end
    m
end   
%%
%Compute estimate (mean and std of realizations)
f_0 = NaN(length(sim),2);
for i = 1:length(sim)
    f_0(i,1) = nanmean(payoff(:,i))';
    f_0(i,2) = nanstd(payoff(:,i))';
end
%%
sim = 5000:5000:50000;
plot(1:length(sim), ones(length(sim),1).*BS_f_v)
hold on
errorbar(1:length(sim), f_0(:,1), 1.96.*f_0(:,2)./sqrt(sim'),'rx'); %Assuming normal distribution for estimate values.
title('MC European Put Price Estimate & 95% Confidence Intervals');
xlabel('Number of Realizations');
set(gca,'XTick', 1:length(sim), 'XTickLabel', sim);
ylabel('Option Price Estimate');




%% MC Price Sensitivity Analysis (Volatility) 
% Trials and realizations are fixed such that std of estimate is low
sim = 30000;
vol_ = 0.1:0.1:1;

s_T_vol = NaN(sim, length(vol_));
payoff_vol = NaN(sim, length(vol_));
for m = 1:length(vol_)
        for j = 1:sim
            %Compute stock price at T
            s_T_vol(j,m) = s_0*exp((r-0.5*vol_(m)^2)*T + vol_(m)*sqrt(T)*normrnd(0,1));
            %Compute discounted payoff
            if(k - s_T_vol(j,m) > 0)
                payoff_vol(j,m) = exp(-r*T)*(k - s_T_vol(j,m));
            else
                payoff_vol(j,m) = 0;
            end
        end
    m
end   

%%
%Compute mean and std of estimate for varying number of trials
f_0_vol = NaN(length(vol_), 2);
for i = 1:length(vol_)
    f_0_vol(i,1) = nanmean(payoff_vol(:,i))';
    f_0_vol(i,2) = nanstd(payoff_vol(:,i))';
end
%%
vol_ = 0.1:0.1:1;

errorbar(1:length(vol_), f_0_vol(:,1), 1.96.*f_0_vol(:,2)./sqrt(sim),'rx'); %Assuming normal distribution for estimate values.
title('MC European Put Price Estimate with Changing Volatility and 95% Confidence Intervals');
xlabel('Volatility');
set(gca,'XTick', 1:length(vol_), 'XTickLabel', vol_);
ylabel('Option Price Estimate');


%% MC Price Sensitivity Analysis (Strike Price) 
% Trials and realizations are fixed such that std of estimate is low
sim = 30000;
k_ = 89:1:109;

s_T_k = NaN(sim,1);
 for j = 1:sim
            %Compute stock price at T
            s_T_k(j) = s_0*exp((r-0.5*vol^2)*T + vol*sqrt(T)*normrnd(0,1));
 end
 
payoff_k = NaN(sim, length(k_));
for m = 1:length(k_)
        for j = 1:sim
            %Compute discounted payoff
            if(k_(m) - s_T_k(j) > 0)
                payoff_k(j,m) = exp(-r*T)*(k_(m) - s_T_k(j));
            else
                payoff_k(j,m) = 0;
            end
        end
    m
end   

%%
%Compute mean and std of estimate for varying number of trials
f_0_k = NaN(length(k_), 2);
for i = 1:length(k_)
    f_0_k(i,1) = nanmean(payoff_k(:,i))';
    f_0_k(i,2) = nanstd(payoff_k(:,i))';
end
%%
k_ = 89:1:109;

errorbar(1:length(k_), f_0_k(:,1), 1.96.*f_0_k(:,2)./sqrt(sim),'rx'); %Assuming normal distribution for estimate values.
title('MC European Put Price Estimate with Changing Strike Price and 95% Confidence Intervals');
xlabel('Strike Price');
set(gca,'XTick', 1:length(k_), 'XTickLabel', k_);
ylabel('Option Price Estimate');


