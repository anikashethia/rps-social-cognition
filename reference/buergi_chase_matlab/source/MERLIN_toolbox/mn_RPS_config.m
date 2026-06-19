function task = mn_RPS_config()
%
%
%

% task
task.fxn = @mn_RPS_task;
task.is_dynamic = true;
task.n_trials = 40;
task.n_blocks = 6;

% game
task.game.strat_space   = 3;
task.game.win           = 1;
task.game.loss          = -1;
task.game.tie           = 0;
task.game.focal_own     = NaN;
task.game.focal_other   = NaN;
task.game.bonus_A       = 0;
task.game.bonus_B       = 0;

% bot
task.bot.params.alpha = 0.9;
task.bot.params.beta = 10;
% task.bot.levels   = [0 1 2 0 1 2 0 1 2];

% create random bot level sequence (without repetitions of levels, and not starting with k=2)
levels = 0:2;
n_levels = numel(levels);
generate_seq = 1;
while generate_seq
    block_level_k = [levels(randperm(n_levels)) levels(randperm(n_levels)) levels(randperm(n_levels))];
    level_changes = diff(block_level_k);
    if block_level_k(1) < 2 && all(level_changes(n_levels:n_levels:task.n_blocks-1) ~= 0)
        generate_seq = 0;
    end
end
task.bot.levels = block_level_k;

end