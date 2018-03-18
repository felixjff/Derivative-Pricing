%Purpose: Implement the solution to Black-Scholes PDE under several schemes 
% and compare it to the solution of the analytical formula in the 
% context of a European Call option. 

%% Case 1: In the money call option
r = 0.04;
sd = 0.30;
s_0 = [100, 110, 120]; 
k = 110;
T = 1;
m = 100;
n = 100;

for i = 1:1:3
    v_CN(i) = BS_PDE_CN_EUROption(s_0(i), k, r, sd, T, n, m, 'Call');
    v_A(i) =  Black_Scholes_EUCall(s_0(i),k,r,sd,T);
end

%% Price as a function of initial value 
s_0 = 30:1:120;
for i = 1:1:length(s_0)
    v_CN_s(i) = BS_PDE_CN_EUROption(s_0(i), k, r, sd, T, n, m, 'Call');
end

plot(s_0,v_CN_s);
title('Option Price (Crank-Nickolson)');
xlabel('Initial Value');
ylabel('Option Value');

%% Convergence in space
s_0 = 110
n = 10:1:100;
m = 10:1:500;
v_CN_m = NaN(length(n),length(m));
for j = 1:1:length(n)
for i = 1:1:length(m)
    v_CN_m(j,i) = BS_PDE_CN_EUROption(s_0, k, r, sd, T, n(j), m(i), 'Call');
    i
end
end

%% Convergence in time
n = 10:1:500;
m = 10;
v_CN_m = NaN(length(m),1);
for i = 1:1:length(m)
    v_CN_n(i) = BS_PDE_CN_EUROption(s_0, k, r, sd, T, n, m(i), 'Call');
    i
end


%%
plot(m,v_CN_m);
title('Option Price (Crank-Nickolson)');
xlabel('Grid Size in Space');
ylabel('Option Value');