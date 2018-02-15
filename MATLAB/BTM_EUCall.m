%Purpose: Adaptable Binomial-Tree model for European Call Option pricing. 

function [output] = BTM_EUCall(s_0,k,r,vol,T,N)

%% Tree Parameters
u = exp(vol*sqrt(T/N));
d = exp(-vol*sqrt(T/N));
p = (exp(r*(T/N)) - d)/(u - d);

%% Backward Iteration

%For computational efficiency, start at end and iterate backwards
   %Create vector with stock values at time N
   js = (N+1):-1:1;
   s_v = (s_0.*u.^(js-1) .* d.^(N+1-js))';
   
   %Compute option value at T
   for j = 1:(N+1)
     if s_v(j)-k>0
        f_v(j) = s_v(j)-k;
     else
        f_v(j) = 0;
     end
   end

   %%
for i = (N+1):-1:2  
   %Create space for option price at T-1
   f_v_ = zeros(i-1,1);
   
   %Compute option prices at T-1 in each node
   for j = i:-1:2
     f_v_(j-1) = (p*f_v(j-1) + (1-p)*f_v(j))*exp(-r*(1/N));
   end
    
   f_v = f_v_;
end

output = f_v;
