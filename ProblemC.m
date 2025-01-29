%Problem C

clc; clear; close all;

% Define parameters
params.r = 0.04;       % Interest rate
params.d = 0.95;
params.A = 1;
params.a = 0.3;
params.p = 0.1;
params.S = 1000;       % Grid size
params.maxit = 10000;  % Maximum iterations
params.crit = 1e-6;    % Convergence criterion
params.Delta = 1000;   % Time step

% Solve the model
[v, I, k, dist] = OneSecGrowth_FDM_fun(params);

% Value Function Plot
figure('Position', [100, 100, 900, 400])

subplot(1,2,1)
plot(k, v, 'LineWidth', 2)
grid on
set(gca, 'FontSize', 12)
xlabel('k')
ylabel('v(k)')
title('Value Function')

subplot(1,2,2)
plot(k, I, 'LineWidth', 2)
grid on
set(gca, 'FontSize', 12)
xlabel('k')
ylabel('I')
title('Policy Function')

function [v, I, k, dist] = OneSecGrowth_FDM_fun(params)
    % Solve continuous-time one sector growth model using finite difference method
    % Input: params struct with model parameters
    % Output: value function (v), consumption (c), capital grid (k), convergence path (dist)
    
    % Extract parameters
    r = params.r;
    d = params.d;
    a = params.a;
    p = params.p;
    S = params.S;
    maxit = params.maxit;
    crit = params.crit;
    Delta = params.Delta;
    
    
    % Setup capital grid
    kmin = 0.1;
    kmax = 10;
    k = linspace(kmin, kmax, S)';
    dk = (kmax-kmin)/(S-1);
    
    % Initialize arrays
    dVf = zeros(S,1);
    dVb = zeros(S,1);
    I = zeros(S,1);
    dist = zeros(maxit,1);
    
    % Initial guess for value function
    tv = k.^a;
    
    % Main iteration loop
    for n = 1:maxit
        v = tv;
        
        % Forward difference
        dVf(1:S-1) = diff(v)/dk;
        dVf(S) = p*((d*kmax)/kmax - d);
        
        % Backward difference
        dVb(2:S) = diff(v)/dk;
        dVb(1) = p*((d*kmin)/kmin - d);
        
        % Consumption and savings
        If = (dVf./p + d).*k;
        muf = If - d.*k;
        Ib = (dVb./p + d).*k;
        mub = Ib - d.*k;
        
        % Steady state values
        I0 = d.*k;
        dV0 = p*(I0./k-d);
        
        % Upwind scheme
        If = muf > 0;
        Ib = mub < 0;
        I0 = (1-If-Ib);
        dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0;
        
        % Update consumption and utility
        I = (dV_Upwind./p + d).*k;
        u = k.^a - adjc(I,k,params);
        
        % Construct sparse transition matrix
        X = -min(mub,0)/dk;
        Y = -max(muf,0)/dk + min(mub,0)/dk;
        Z = max(muf,0)/dk;
        Amat = spdiags(Y,0,S,S) + spdiags(X(2:S),-1,S,S) + spdiags([0;Z(1:S-1)],1,S,S);
        %% 
        
        % Check transition matrix
        if max(abs(sum(Amat,2))) > 1e-12
            error('Improper Transition Matrix');
        end
        
        % Solve system of equations
        B = (r + 1/Delta)*speye(S) - Amat;
        b = u + v/Delta;
        tv = B\b;
        
        % Check convergence
        Vchange = tv - v;
        dist(n) = max(abs(Vchange));
        
        if dist(n) < crit
            fprintf('Value Function Converged, Iteration = %d\n', n);
            dist = dist(1:n);
            break;
        end
    end
end

function cost = adjc(I,k,params)
     cost = params.p/2.*(I./k -params.d).^2.*k;
end

