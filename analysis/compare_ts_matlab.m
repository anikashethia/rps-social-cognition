%
% compare_ts_matlab.m
%
% Tests whether mn_RPS_task.m math matches agents.ts.
% Reimplements mn_RPS_task.m probability + update equations directly,
% runs them on the same fixed 40-trial choice sequence as compare_ts_matlab.mjs,
% and writes matlab_probs.json.
%
% Run from buergi_chase/ in MATLAB:
%   cd('~/github_tasks/buergi_chase')
%   run('~/github_tasks/rps-social-cognition/analysis/compare_ts_matlab.m')
%
% Then run: python3 analysis/compare_ts_matlab.py
%

ALPHA = 0.9;
BETA  = 10;

% Payoff matrix [myAction, opponentAction], 1-indexed, R=1/P=2/S=3
% Matches comp_paymatrix output: both pi_subj and pi_bot are identical for symmetric RPS.
% Verified from comp_paymatrix.m: pay_matrix(i,j) = payoff for row player playing i vs j.
payoff = [ 0, -1,  1; ...
            1,  0, -1; ...
           -1,  1,  0];

% Softmax — matches mn_RPS_task.m softmax_fxn (no max subtraction; same result for these inputs)
softmax_fn = @(v) exp(BETA * v(:)') / sum(exp(BETA * v(:)'));  % always returns 1×3 row

% Fixed 40-trial sequence: [p_choice, bot_choice], 1-indexed (R=1, P=2, S=3)
% Identical to compare_ts_matlab.mjs SEQ.
SEQ = [ ...
  1,2; 1,3; 2,1; 3,2; 1,1; 2,3; 3,1; 1,2; 2,2; 3,3; ...
  1,3; 2,1; 3,2; 1,1; 2,3; 3,3; 1,2; 2,1; 3,3; 1,2; ...
  2,2; 3,1; 1,3; 2,2; 3,1; 1,1; 2,3; 3,2; 1,3; 2,1; ...
  3,3; 1,2; 2,1; 3,1; 1,3; 2,2; 3,3; 1,1; 2,3; 3,2; ...
];

n_trials = size(SEQ, 1);
levels   = [0, 1, 2];
results  = struct([]);

for li = 1:numel(levels)
    level  = levels(li);
    f_bot  = ones(3,1)/3;   % bot's own attraction tracker  (= agents.ts attr)
    f_subj = ones(3,1)/3;   % participant's attraction tracker (= agents.ts pAttr)

    for t = 1:n_trials
        p_choice   = SEQ(t,1);
        bot_choice = SEQ(t,2);

        % Compute probability vector BEFORE update — matches agents.ts choose() call order.
        % Equations from mn_RPS_task.m lines 86-98 (Version from CHASE paper):
        %   k=0: softmax(f_bot, β)
        %   k=1: softmax(π × f_subj, β)                         (raw f_subj, no initial softmax)
        %   k=2: bot_pred = softmax(π × f_bot, β)
        %        p = softmax(π × bot_pred', β)
        if level == 0
            p = softmax_fn(f_bot);
        elseif level == 1
            p = softmax_fn(payoff * f_subj);
        else  % level == 2
            bot_pred = softmax_fn(payoff * f_bot);
            p        = softmax_fn(payoff * bot_pred');
        end

        n = numel(results) + 1;
        results(n).trial      = t;
        results(n).level      = level;
        results(n).p_choice   = p_choice;
        results(n).bot_choice = bot_choice;
        results(n).probs      = p;
        results(n).attr       = f_bot';
        results(n).pAttr      = f_subj';

        % Deterministic update — mn_RPS_task.m lines 142-150:
        %   f_mat_subj(t,:) = (1-α)*f_mat_subj(t-1,:) + α*one_hot(choice_own(t))
        %   f_mat_bot(t,:)  = (1-α)*f_mat_bot(t-1,:)  + α*one_hot(choice_other(t))
        ind_bot           = zeros(3,1); ind_bot(bot_choice)  = 1;
        ind_subj          = zeros(3,1); ind_subj(p_choice)   = 1;
        f_bot             = (1-ALPHA)*f_bot  + ALPHA*ind_bot;
        f_subj            = (1-ALPHA)*f_subj + ALPHA*ind_subj;
    end
end

% Save as JSON
out_path = fullfile(getenv('HOME'), 'github_tasks', 'rps-social-cognition', 'analysis', 'matlab_probs.json');
json_str = jsonencode(results);
fid = fopen(out_path, 'w');
fwrite(fid, json_str, 'char');
fclose(fid);
fprintf('Written %d records → %s\n', numel(results), out_path);
