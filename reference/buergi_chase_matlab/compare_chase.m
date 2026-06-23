% Cross-validation script: run the original CHASE model on a fixed trial
% sequence with fixed parameters, to compare trial-by-trial outputs against
% the Python reimplementation in models/chase.py.
%
% Run this in MATLAB Online with the uploaded `source/` folder (and its
% MERLIN_toolbox subfolder) on the path.

addpath(genpath('source'));

% ── Model config: CH architecture, kappa fitted up to 3, RW-freq learning ──
model = BAKR_2024_CHASE_config('CH','fitted',3,'RW-freq');
disp('Parameter order:');
disp({model.params.name});

% ── Fixed trial sequence (1=Rock, 2=Paper, 3=Scissors) ─────────────────────
data.choice_own   = [1;2;3;1;2;3;1;1;2;3;2;1];
data.choice_other = [2;2;2;1;1;1;3;3;3;2;1;3];
data.missing      = zeros(12,1);
data.n_trials     = 12;
data.trial        = (1:12)';  % no block resets mid-sequence

% ── Fixed parameters, in the order printed above: beta, lambda, gamma, alpha, kappa ──
curr_params = [3.0; 1.5; 2.0; 0.3; 3];

[negLL, states, data] = BAKR_2024_CHASE_model(data, model, curr_params, 'fit');

fprintf('negLL: %.6f\n\n', negLL);

disp('lik (P(chosen action) per trial):');
disp(states.lik');

disp('SV (subjective value per trial):');
disp(states.subj_SV');

disp('APE (action prediction error per trial):');
disp(states.subj_APE');

disp('KL_div / BU (belief update per trial):');
disp(states.subj_KL_div');

disp('beliefs (PRIOR going into each trial, rows=trials, cols=levels 0..kappa-1):');
disp(states.beliefs);
