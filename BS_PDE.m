params.Smin = 0.4;
params.Smax = 1000;
params.T = 1;
params.sigma = 0.4;
params.r = 0.02;
params.d = 0.0;
params.K = 10;
params.grid_s = 1000;
params.grid_t = 4000;


[surf, S_vals, t_vals] = solve_BSPDE(params);


figure;

[S_grid, T_grid] = meshgrid(S_vals, t_vals);
mesh(S_grid, T_grid, surf);
xlabel('S');
ylabel('t');
zlabel('Option Value');
title('Black-Scholes PDE solution (Implicit Method)');

function [surf, S, t_vals] = solve_BSPDE(params)
    

    M = params.grid_s;
    N = params.grid_t;
    S = linspace(params.Smin, params.Smax, M)';
    t_vals = linspace(0, params.T, N);
    dt = 1/N;
    surf = zeros(N+1, M+1);

    surf(end,:) = max(S - params.K);
    surf(:,1) = 0;
    surf(:,end) = params.Smax - params.K;
    

    a = @(j)1/2*(params.r-params.d).*j.*dt - 1/2*params.sigma^2.*j.^2*dt;
    b = @(j) 1+params.sigma^2.*j.^2.*dt + params.r.*dt;
    c = @(j) -1/2.*(params.r-params.d).*j.*dt - 1/2*params.sigma.^2.* j.^2.*dt;

    A = diag(a(2:M),-1)+diag(b(1:M))+diag(c(1:M-1),1);
    
    for i = N:-1:1

        v = surf(i+1, 1:M)';

        v(1) = v(1) - a(1)* surf(i, 1);

        v(end) = v(end) - c(M+1)* surf(i, M+1);

        surf(i, 2:M) = A\v;
    end
end