N = 1000;
M = 1000;
Smin = 0.4;
Smax = 1000;
T = 1;
K = 10;
volatility = 0.2;
r = 0.02;
d = 0;
is_call = true;

[t_vals, S_vals, surf] = bs_implicit( ...
    N, M, Smin, Smax, T, K, volatility, r, d, is_call);

figure;
mesh(S_vals, t_vals, surf);
xlabel('Asset Price S');
ylabel('Time t');
zlabel('Option Value');
title('Black-Scholes Implicit Method');

function [t_vals,S_vals,surf] = bs_implicit( ...
    N,M,Smin,Smax,T,K,volatility,r,d,is_call)

    surf = zeros(N+1, M+1);
    dt = 1 / N;
    dS = (Smax - Smin) / M;
    t_vals = 0 : dt : T;
    S_vals = Smin : dS : Smax;

    surf(:, 1)   = 0;
    surf(:, end) = Smax - K;
    surf(end, :) = payoff(S_vals, K, is_call);

    a = @(j)  0.5*(r-d).*j*dt - 0.5*volatility.^2 * j.^2 * dt;
    b = @(j)  1 + volatility.^2 * j.^2 * dt + r*dt;
    c = @(j) -0.5*(r-d)*j*dt - 0.5*volatility.^2 * j.^2 * dt;

    for i = N:-1:1
        A = diag(a(2:M-1), -1) + diag(b(1:M-1)) + diag(c(1:M-2), 1);
        v = surf(i+1, 2:M)';
        v(1)   = v(1)   - a(1)*surf(i,1);
        v(end) = v(end) - c(M+1)*surf(i, M+1);
        surf(i, 2:M) = A \ v;
        surf(i, 2:M) = max(surf(i, 2:M), payoff(S_vals(2:M), K, is_call));
    end
end

function val = payoff(S, K, is_call)
    if is_call
        val = max(S - K, 0);
    else
        val = max(K - S, 0);
    end
end
