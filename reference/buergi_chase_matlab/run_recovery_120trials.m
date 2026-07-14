%
% Parameter recovery test: 120-trial CHASE fit (3 blocks × 40 trials, k=0/1/2)
%
% Simulates a synthetic participant with known CHASE parameters playing against
% a bot at CHASE levels 0, 1, and 2 (one block each, 40 trials per block),
% then fits CHASE to recover those parameters using Buergi's mn_fit pipeline.
%
% Matches our task design: one agent, 3 blocks, each at a different k level.
%
% Run from the buergi_chase_matlab/ directory:
%   cd reference/buergi_chase_matlab
%   run_recovery_120trials
%

addpath(genpath('source'));
rng(42);  % reproducible

%% Task config: 3 blocks, 40 trials each, bot levels [0, 1, 2]

task          = mn_RPS_config();
task.n_blocks = 3;
task.bot.levels = [0, 1, 2];  % one permutation of all three levels

%% CHASE model config

model         = BAKR_2024_CHASE_config('CH', 'fitted', 3, 'RW-freq');
model.sim_fxn = model.loglik_fxn;  % required for mn_sim

% Parameter order: beta, lambda, gamma, alpha, kappa
fprintf('Parameter order: ');
fprintf('%s  ', model.params.name);
fprintf('\n\n');

%% True parameters — edit these to test different regimes

true_params = [10.0;   % beta   — softmax temperature (bot default is 10)
                1.0;   % lambda — loss sensitivity (1 = neutral)
                2.0;   % gamma  — sensitivity to opponent level evidence
                0.3;   % alpha  — attraction learning rate
                  2];  % kappa  — max reasoning depth (0, 1, 2, or 3)

fprintf('True parameters:\n');
for i = 1:numel(model.params)
    fprintf('  %-8s = %.3f\n', model.params(i).name, true_params(i));
end
fprintf('\n');

%% Simulate

fprintf('Simulating 120 trials...\n');
sim      = mn_sim(task, model, true_params);
sim_data = sim.subj.data;   % already has n_blocks=1 (set by BAKR_2024_CHASE_model)
sim_data.subjID = 1;

fprintf('  Simulated %d trials (%d blocks)\n', sim_data.n_trials, task.n_blocks);
fprintf('  Bot levels: %s\n', num2str(task.bot.levels));
fprintf('  Participant win rate: %.2f\n\n', mean(sim_data.score_own > 0));

%% Fit

fprintf('Fitting CHASE model...\n');
fit = mn_fit(sim_data, model);

%% Report

fprintf('\n=== Recovery results ===\n');
fprintf('%-8s   %8s   %8s   %8s\n', 'param', 'true', 'recovered', 'error');
fprintf('%s\n', repmat('-', 1, 42));
param_names = {model.params.name};
for i = 1:numel(param_names)
    name      = param_names{i};
    true_val  = true_params(i);
    rec_val   = fit.subj(1).params.(name);
    err       = rec_val - true_val;
    fprintf('%-8s   %8.3f   %8.3f   %+8.3f\n', name, true_val, rec_val, err);
end
fprintf('\nnegLL = %.4f\n', fit.subj(1).optim.negLL);
fprintf('AIC   = %.4f\n', fit.subj(1).optim.AIC);
