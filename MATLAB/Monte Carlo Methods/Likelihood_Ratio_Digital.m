%Purpose: Determine the hedge parameter (delta) of a Digital option using Monte
%Carlo simulation and the likelihood ratio method. 

%% Parameters
T = 1;
k = 99;
r = 0.06;
s_0 = 100;
vol = 0.2;

%% Simulate stock price at maturity
sim = 100000;
s_T = NaN(sim,1);
z = normrnd(0,1,sim,1);

s_T = s_0*exp((r-0.5*vol^2)*T +  vol*sqrt(T)*z);


%% Determine payoff
p_d = NaN(sim,1);
for i = 1:sim
   if s_T(i) > k
       p_d(i) = 1;
   else
       p_d(i) = 0;
   end
end

%% Compute delta

d_detal = mean(exp(-r*T)*p_d.*z./(vol*s_0*sqrt(T)));
