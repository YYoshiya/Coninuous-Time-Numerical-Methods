function [v, c, k, dist] = OneSecGrowth_FDM_fun(params)
    % Solve continuous-time one sector growth model using finite difference method
    % Input: params struct with model parameters
    % Output: value function (v), consumption (c), capital grid (k), convergence path (dist)
    
    % Extract parameters
    s = params.s;
    a = params.a;
    d = params.d;
    r = params.r;
    I = params.I;
    maxit = params.maxit;
    crit = params.crit;
    Delta = params.Delta;
    
    % Compute steady state capital
    kss = (a/(r+d))^(1/(1-a));
    
    % Setup capital grid
    kmin = 0.001*kss;
    kmax = 2*kss;
    k = linspace(kmin, kmax, I)';
    dk = (kmax-kmin)/(I-1);
    
    % Initialize arrays
    dVf = zeros(I,1);
    dVb = zeros(I,1);
    c = zeros(I,1);
    dist = zeros(maxit,1);
    
    % Initial guess for value function
    tv = (k.^a).^(1-s)/(1-s)/r;
    
    % Main iteration loop
    for n = 1:maxit
        v = tv;
        
        % Forward difference
        dVf(1:I-1) = diff(v)/dk;
        dVf(I) = (kmax^a - d*kmax)^(-s);
        
        % Backward difference
        dVb(2:I) = diff(v)/dk;
        dVb(1) = (kmin^a - d*kmin)^(-s);
        
        % Consumption and savings
        cf = dVf.^(-1/s);
        muf = k.^a - d.*k - cf;
        cb = dVb.^(-1/s);
        mub = k.^a - d.*k - cb;
        
        % Steady state values
        c0 = k.^a - d.*k;
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