clc; clear; close all;

% Define parameters
params.s = 2;          % Risk aversion
params.a = 0.3;        % Capital share
params.d = 0.05;       % Depreciation rate
params.r = 0.04;       % Interest rate
params.I = 10000;      % Grid size
params.maxit = 10000;  % Maximum iterations
params.crit = 1e-6;    % Convergence criterion
params.Delta = 1000;   % Time step