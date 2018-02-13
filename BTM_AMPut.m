%Purpose: Adaptable Binomial-Tree model for American Put Option pricing. 

function [output] = BTM_AMPut(s_0,k,r,vol,T,N)

%% Tree Parameters
u = exp(vol*sqrt(T/N));
d = exp(-vol*sqrt(T/N));
p = (exp(r*(T/N)) - d)/(u - d);

%% Backward Iteration

%For computational efficiency, start at end and iterate backwards
   %Create vector with stock values at time N
   js = (N+1):-1:1;
   s_v = (s_0.*u.^(js-1) .* d.^(N+1-js))';
   
   %Compute (exercise) option value at T (algorithm initialization)
   for j = 1:(N+1)
     if k-s_v(j)>0
        f_v_A(j) = k-s_v(j);
     else
        f_v_A(j) = 0;
     end
   end

   %% Perform backward iteration
for i = (N+1):-1:3  
   %Create storage space for no-arbitrage option value at T-1
   f_v_NA = zeros(i-1,1);
   f_v_A_ = zeros(i-1,1);
   
   %Compute no-arbitrage option value at T-1 in each node
   for j = i:-1:2
     f_v_NA(j-1) = (p*f_v_A(j-1) + (1-p)*f_v_A(j))*exp(-r*(1/N));
   end
   
   %Compute exercise option value at T-1 in each node
   js = (i-1):-1:1;
    %Compute stock prices at T-1
   s_v = (s_0.*u.^(js-1) .* d.^(i-1-js))';
    %Compute option exercise value at T-1
   for j = 1:(i-1)
     if k-s_v(j)>0
        f_v_A_(j) = k-s_v(j);
     else
        f_v_A_(j) = 0;
     end
   end
   
   %Compute value of amercan option at T-1: Greatest of exercise value or
   %no-arbitrage value
   for j = 1:(i-1)
     if f_v_NA(j)>f_v_A_(j)
        f_v_A_(j) = f_v_NA(j);
     end
   end
    
   f_v_A = f_v_A_;
end

%%
%Given the teh value of the american option at both nodes in the first
%step, compute the value at t = 0 using no-arbitrage pricing. 
AM_price = (p*f_v_A(1) + (1-p)*f_v_A(2))*exp(-r*(1/N));

output = AM_price;
