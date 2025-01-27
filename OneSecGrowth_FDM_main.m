clc; clear; close all;


% Define parameters
params.s = 2;          % Risk aversion
params.a = 0.3;        % Capital share
params.d = 0.05;       % Depreciation rate
params.r = 0.05;       % Interest rate
params.I = 10000;      % Grid size
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