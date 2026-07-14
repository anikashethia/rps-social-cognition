%
% load_for_buergi.m
%
% Reads the CSV produced by export_for_matlab.py and saves behavioral_data.mat
% in the format expected by BAKR_2024_run_model_fitting.m.
%
% Usage (run from the rps-social-cognition repo root, or adjust paths below):
%
%   buergi_repo = '~/github_tasks/buergi_chase';
%   run('analysis/load_for_buergi.m')
%
% Or just set the two paths below and run this file directly.
%

%% ── Paths ────────────────────────────────────────────────────────────────────

csv_path    = fullfile(fileparts(mfilename('fullpath')), 'buergi_export.csv');
buergi_repo = fullfile('~', 'github_tasks', 'buergi_chase');
out_path    = fullfile(buergi_repo, 'data', 'behavioral_data.mat');

%% ── Load CSV ─────────────────────────────────────────────────────────────────

fprintf('Reading %s ...\n', csv_path);
raw = readtable(csv_path, 'TextType', 'string');

fprintf('  %d rows, %d participants\n', height(raw), numel(unique(raw.subjID)));

%% ── Build MATLAB table in Buergi format ──────────────────────────────────────
%
% Required columns (checked against mn_table2struct / mn_RPS_task.m):
%   subjID        — numeric
%   choice_own    — 1/2/3 or NaN if missing
%   choice_other  — 1/2/3
%   score_own     — +1/-1/0 or NaN if missing
%   bot_level     — 0/1/2
%   missing       — 0/1
%   trial         — 1:40 within each block
%   block         — 1:6
%   condition     — string
%   dataset       — string
%

data = table();
data.subjID       = raw.subjID;
data.choice_own   = raw.choice_own;
data.choice_other = raw.choice_other;
data.score_own    = raw.score_own;
data.bot_level    = raw.bot_level;
data.missing      = raw.missing;
data.trial        = raw.trial;
data.block        = raw.block;
data.condition    = raw.condition;
data.dataset      = raw.dataset;

%% ── Sanity checks ────────────────────────────────────────────────────────────

subjs     = unique(data.subjID);
n_subjs   = numel(subjs);
n_missing = sum(data.missing);

fprintf('\nSanity checks:\n');
fprintf('  Participants : %d\n', n_subjs);
fprintf('  Total trials : %d  (expected %d per participant)\n', height(data), 240);
fprintf('  Missing      : %d  (%.1f%%)\n', n_missing, 100*n_missing/height(data));
fprintf('  Levels       : %s\n', num2str(unique(data.bot_level)'));
fprintf('  choice_own range  : %g – %g\n', min(data.choice_own,[],'omitnan'), max(data.choice_own,[],'omitnan'));
fprintf('  choice_other range: %g – %g\n', min(data.choice_other), max(data.choice_other));

assert(all(ismember(data.bot_level, [0 1 2])),    'bot_level must be 0, 1, or 2');
assert(all(data.choice_other >= 1 & data.choice_other <= 3), 'choice_other out of range');
assert(all(data.missing == 0 | data.missing == 1), 'missing must be 0 or 1');

%% ── Save ─────────────────────────────────────────────────────────────────────

out_dir = fileparts(out_path);
if ~isfolder(out_dir)
    mkdir(out_dir);
end

save(out_path, 'data');
fprintf('\nSaved → %s\n', out_path);
fprintf('Next: open MATLAB in %s and run BAKR_2024_run_model_fitting.m\n', buergi_repo);
