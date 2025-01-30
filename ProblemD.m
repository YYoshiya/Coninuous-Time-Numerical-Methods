params.Smin = 0.4;
params.Smax = 1000;
params.T = 1;
params.sigma = 0.4;
params.r = 0.02;
params.d = 0.0;
params.K = 10;
params.grid_s = 1000;
params.grid_t = 1000;

V = solve_BSPDE(params)

function v = solve_BSPDE(params)
    

    M = params.grid_s;
    N = params.grid_t;
    S = linspace(params.Smin, params.Smax, M + 1)';
    dt = 1/N;
    surf = zeros(N+1, M+1);
    surf(N+1,:) = max(S - params.K);
    

    j = (1:M)';
    a = 1/2*(params.r-params.d).*j.*dt - 1/2*params.sigma^2.*j.^2*dt;
    b = 1+params.sigma^2.*j.*dt + params.r.*dt;
    c = -1/2.*(params.r-params.d).*j.*dt - 1/2*params.sigma^2.* j.^2.*dt;

    
    A = diag(a(2:M-1), -1) + diag(b(2:M)) + diag(c(1:M-2), 1);
    
    for i = N:-1:1

        v = surf(i+1, 2:M)';

        v(1) = v(1) - a(1)* surf(i, 1);

        v(end) = v(end) - c(M)* surf(i, M+1);

        surf(i, 2:M) = A\v;
    end
end