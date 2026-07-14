%
% run_buergi_recovery.m
%
% Replicates Buergi et al.'s supplementary parameter recovery figure exactly:
%
%   1. Load their behavioral_data.mat (fMRI participants)
%   2. Fit CHASE to each participant using mn_fit
%   3. For each participant × kappa (0,1,2): simulate a fake session with
%      their fitted params (alpha,beta,lambda,gamma) + that kappa using mn_sim
%   4. Fit CHASE to each fake session
%   5. Plot true vs recovered — directly comparable to their supplementary figure
%
% Run from the buergi_chase/ directory in MATLAB:
%   cd('~/github_tasks/buergi_chase')
%   run('~/github_tasks/rps-social-cognition/analysis/run_buergi_recovery.m')
%

%% ── Setup ────────────────────────────────────────────────────────────────────

project_folder = cd;  % buergi_chase/
addpath(fullfile(project_folder, 'source'));
addpath(fullfile(project_folder, 'source', 'MERLIN_toolbox'));

out_dir = fullfile('~', 'github_tasks', 'rps-social-cognition', 'analysis');

%% ── Step 1: Load and fit CHASE to Buergi's fMRI participants ─────────────────

fprintf('=== Step 1: Fitting CHASE to Buergi fMRI participants ===\n');

load(fullfile(project_folder, 'data', 'behavioral_data.mat'));
data = mn_table2struct(data, 'subjID', 'remove_redundancy', ...
                       'exceptions', {'choice_own', 'choice_other', 'missing'});

fprintf('  Loaded %d participants\n', numel(data));

model         = BAKR_2024_CHASE_config('CH', 'fitted', 2, 'RW-freq');
model.sim_fxn = model.loglik_fxn;

fprintf('  Fitting CHASE (kappa 0-2)... this may take a few minutes\n\n');
fits = mn_fit(data, model);

fprintf('\n  Done. Fitted %d participants.\n\n', numel(fits.subj));

%% ── Step 2: Parameter recovery — simulate at each kappa level ────────────────

fprintf('=== Step 2: Simulating recovery data ===\n');

task          = mn_RPS_config();
task.n_blocks = 3;

kappas        = 0:2;
n_subj        = numel(fits.subj);
param_names   = {model.params.name};   % {'beta','lambda','gamma','alpha','kappa'}
n_params      = numel(param_names);

true_mat = [];
sim_data = [];

for k = kappas
    fprintf('  Simulating kappa=%d (%d participants)...\n', k, n_subj);
    for i_subj = 1:n_subj

        % Use this participant's fitted params, override kappa
        fitted = fits.subj(i_subj).params;
        curr_params = [
            fitted.beta;
            fitted.lambda;
            fitted.gamma;
            fitted.alpha;
            k               % override kappa
        ];

        % Use the bot level sequence this participant actually experienced
        subj_levels = fits.subj(i_subj).data.bot_level(1:40:end);
        task.bot.levels = subj_levels(1:3);  % one permutation of 3 levels

        % Simulate
        sim        = mn_sim(task, model, curr_params);
        s          = sim.subj.data;
        s.subjID   = (k + 1) * 100 + i_subj;
        s.n_blocks = 1;

        sim_data   = [sim_data; s];
        true_mat   = [true_mat; curr_params'];
    end
end

fprintf('  Total simulations: %d\n\n', size(true_mat, 1));

%% ── Step 3: Fit CHASE to each simulated session ──────────────────────────────

fprintf('=== Step 3: Fitting CHASE to simulated sessions ===\n\n');
fits_rec = mn_fit(sim_data, model);

%% ── Step 4: Extract recovered parameters ─────────────────────────────────────

n_sims   = numel(fits_rec.subj);
rec_mat  = NaN(n_sims, n_params);

for i = 1:n_sims
    for j = 1:n_params
        rec_mat(i, j) = fits_rec.subj(i).params.(param_names{j});
    end
end

%% ── Step 5: Report correlations ──────────────────────────────────────────────

fprintf('\n=== Recovery results ===\n');
fprintf('%-8s   %6s\n', 'param', 'r');
fprintf('%s\n', repmat('-', 1, 18));

r_vals = NaN(1, n_params);
for j = 1:n_params
    ok = ~isnan(true_mat(:,j)) & ~isnan(rec_mat(:,j));
    if sum(ok) > 2
        r_vals(j) = corr(true_mat(ok,j), rec_mat(ok,j));
        fprintf('%-8s   %6.3f\n', param_names{j}, r_vals(j));
    end
end

%% ── Step 6: Plot ─────────────────────────────────────────────────────────────

kappa_true   = true_mat(:, end);   % last column = kappa
kappa_colors = [0.22 0.50 0.82;    % k=0 blue
                0.93 0.60 0.14;    % k=1 orange
                0.17 0.63 0.17];   % k=2 green

fig = figure('Position', [100 100 1200 230], 'Visible', 'off');

for j = 1:n_params
    ax = subplot(1, n_params, j);
    hold on;

    for ki = 1:numel(kappas)
        k_idx = (kappa_true == kappas(ki));
        scatter(true_mat(k_idx, j), rec_mat(k_idx, j), 30, ...
            kappa_colors(ki,:), 'filled', 'MarkerFaceAlpha', 0.75, ...
            'DisplayName', sprintf('\\kappa=%d', kappas(ki)));
    end

    % Identity line
    all_v = [true_mat(:,j); rec_mat(:,j)];
    lims  = [min(all_v,[],'omitnan'), max(all_v,[],'omitnan')];
    pad   = 0.05 * diff(lims);
    lims  = [lims(1)-pad, lims(2)+pad];
    plot(lims, lims, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.2, ...
        'HandleVisibility', 'off');

    xlim(lims); ylim(lims);
    xlabel(sprintf('True %s', param_names{j}), 'FontSize', 9);
    ylabel(sprintf('Recovered %s', param_names{j}), 'FontSize', 9);

    if ~isnan(r_vals(j))
        title(sprintf('%s   r = %.2f', param_names{j}, r_vals(j)), 'FontSize', 9);
    end
    if j == 1
        legend('Location', 'northwest', 'FontSize', 7);
    end
    box on; axis square;
end

sgtitle('Parameter recovery — Buergi fMRI participants (our CHASEBot sequences)', ...
        'FontSize', 10);

out_fig = fullfile(out_dir, 'recovery_figure.png');
exportgraphics(fig, out_fig, 'Resolution', 150);
fprintf('\nFigure saved → %s\n', out_fig);

%% ── Save ─────────────────────────────────────────────────────────────────────

out_mat = fullfile(out_dir, 'recovery_results.mat');
save(out_mat, 'fits', 'fits_rec', 'true_mat', 'rec_mat', 'param_names', 'r_vals');
fprintf('Results saved → %s\n', out_mat);
