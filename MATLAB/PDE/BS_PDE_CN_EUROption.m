%Purpose: Price a European call option using Crank-Nickelson scheme to
%approximate Black-Scholes PDE. 

function [output] = BS_PDE_CN_EUROption(s_0,k,r,sd,T, n, m, type)

%Define the step sizes
dt = T/n;
m = m + mod(m,2); %ensures m is even.
    %Set stock price limits (max and min based on common practices)
    x_max = log(s_0) + 3*sd*sqrt(T)*1;
    x_min = log(s_0) + 3*sd*sqrt(T)*-1;
dx = (x_max - x_min)/m;

x_seq = x_min + (0:1:m)*dx;   %vector, m+1 elements

f = NaN(n+1,m+1);

if strcmp(type,'Call')
  f(n+1,:) = max(exp(x_seq)-k, 0);
elseif strcmp(type,'Put')
  f(n+1,:) = max(k-exp(x_seq), 0);
end

  p_u = -dt/4*(sd^2/dx^2 + (r - 1/2*sd^2)/dx);
  p_m = 1 + dt*(sd^2/(2*dx^2) + r/2);
  p_d = -dt/4*(sd^2/dx^2 - (r - 1/2*sd^2)/dx);
  
  diag_1 = [zeros(m+1,1)'; horzcat(diag([p_u*ones(m-1,1);1]),zeros(m,1))];
  diag_2 = diag([ 1; p_m*ones(m-1,1); -1]);
  diag_3 =  [horzcat(zeros(m,1), diag([-1; p_d*ones(m-1,1)])); zeros(m+1,1)'];
  tridiag = diag_1 + diag_2 + diag_3;
  
  A = tridiag;
  
  B = diag(2*ones(m+1,1)) - tridiag; %Note: 1st & last rows corrected later.
  
  for i = n:-1:1
    rhs = B*f(i+1,(m+1):-1:1)';
    if strcmp(type,'Call')
      rhs(1) = exp(x_seq(m+1)) - exp(x_seq(m));
      rhs(m+1) = 0;
    elseif strcmp(type,'Put')
      rhs(1) = 0;
      rhs(m+1) = exp(x_seq(m+1)) - exp(x_seq(m));  
    end
    f(i,(m+1):-1:1) = A\rhs;
  end
  

output = f(1,1 + m/2);