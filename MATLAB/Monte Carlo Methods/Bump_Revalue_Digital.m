%Purpose: Determine the hedge parameter (delta) of a Digital option using Monte
%Carlo simulation and the bump-and-revalue method. 

%% Parameters
T = 1;
k = 99;
r = 0.06;
s_0 = 100;
vol = 0.2;

%%
seed1 = 10;

sim = 100000;
bump = 0.01*s_0;
s_0_bump = NaN(length(bump) + 1,1);
s_0_bump(1) = s_0;
s_0_bump(2) =  s_0 + bump(1);   %First bump
s_T_bump = NaN(sim,length(bump) + 1);
for i = 1:(length(bump)+1)
  for j = 1:sim
    %Compute stock price at T
    s_T_bump(j,i) = s_0_bump(i)*exp((r-0.5*vol^2)*T + vol*sqrt(T)*normrnd(0,1));
  end
end

%% 
%Value - Revalue
payoff_bump = NaN(sim, length(bump)+1);
for i = 1:(length(bump)+1)
 for j = 1:sim
      %Compute discounted payoff
      if(s_T_bump(j,i) - k > 0)
         payoff_bump(j,i) = exp(-r*T)*1;
      else
         payoff_bump(j,i) = 0;
      end
 end
end
f_v = NaN(length(bump)+1, 3);
for i = 1:length(bump)+1
 f_v(:,1) = mean(payoff_bump)';
 f_v(:,2) = std(payoff_bump)';
 f_v(:,3) = 1.96.*f_v(:,2)./sqrt(sim);
end

%%
%Sensitivity based on Euler scheme
HP_MC = NaN(length(bump),2);
for i = 2:(length(bump)+1)
   HP_MC(i-1,1) = (f_v(i,1) - f_v(1,1))/bump(i-1); 
   HP_MC(i-1,2) = f_v(i,3) + f_v(1,3);
end
