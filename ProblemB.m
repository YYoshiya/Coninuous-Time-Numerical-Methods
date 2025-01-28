%Problem B

clc; clear; close all;

% Define parameters
params.s = 2;          % Risk aversion  
params.r = 0.04;       % Interest rate
params.R = 0.04;
params.w = 1;          % wage
params.I = 1000;       % Grid size
params.maxit = 10000;  % Maximum iterations
params.crit = 1e-6;    % Convergence criterion
params.Delta = 1000;   % Time step

% Solve the model
[v, c, k, dist] = OneSecGrowth_FDM_fun(params);

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
plot(k, c, 'LineWidth', 2)
grid on
set(gca, 'FontSize', 12)
xlabel('k')
ylabel('c(k)')
title('Policy Function')

function [v, c, a, dist] = OneSecGrowth_FDM_fun(params)
    % Solve continuous-time one sector growth model using finite difference method
    % Input: params struct with model parameters
    % Output: value function (v), consumption (c), capital grid (k), convergence path (dist)
    
    % Extract parameters
    s = params.s;
    r = params.r;
    R = params.R;
    w = params.w;
    I = params.I;
    maxit = params.maxit;
    crit = params.crit;
    Delta = params.Delta;
    
    
    % Setup capital grid
    amin = 0.1;
    amax = 10;
    a = linspace(amin, amax, I)';
    dk = (amax-amin)/(I-1);
    
    % Initialize arrays
    dVf = zeros(I,1);
    dVb = zeros(I,1);
    c = zeros(I,1);
    dist = zeros(maxit,1);
    
    % Initial guess for value function
    tv = (a.^a).^(1-s)/(1-s)/r;
    
    % Main iteration loop
    for n = 1:maxit
        v = tv;
        
        % Forward difference
        dVf(1:I-1) = diff(v)/dk;
        dVf(I) = (w + R.*amax)^(-s);
        
        % Backward difference
        dVb(2:I) = diff(v)/dk;
        dVb(1) = (w + R.*amin)^(-s);
        
        % Consumption and savings
        cf = dVf.^(-1/s);
        muf = w + R.*a - cf;
        cb = dVb.^(-1/s);
        mub = w + R.*a - cb;
        
        % Steady state values
        c0 = w + R.*a;
        dV0 = c0.^(-s);
        
        % Upwind scheme
        If = muf > 0;
        Ib = mub < 0;
        I0 = (1-If-Ib);
        dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0;
        
        % Update consumption and utility
        c = dV_Upwind.^(-1/s);
        u = c.^(1-s)/(1-s);
        
        % Construct sparse transition matrix
        X = -min(mub,0)/dk;
        Y = -max(muf,0)/dk + min(mub,0)/dk;
        Z = max(muf,0)/dk;
        A = spdiags(Y,0,I,I) + spdiags(X(2:I),-1,I,I) + spdiags([0;Z(1:I-1)],1,I,I);
        
        % Check transition matrix
        if max(abs(sum(A,2))) > 1e-12
            error('Improper Transition Matrix');
        end
        
        % Solve system of equations
        B = (r + 1/Delta)*speye(I) - A;
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