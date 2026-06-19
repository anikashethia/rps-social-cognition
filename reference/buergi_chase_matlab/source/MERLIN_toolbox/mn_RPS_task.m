function [data,env] = mn_RPS_task(data,task,type,env,iTrial)
%
% Rock-Paper-Scissors task
%

switch type
    
    case 'init'

        n_trials = task.n_trials;
        n_actions = task.game.strat_space;

        % payoff matrices
        [env.pi_subj,env.pi_bot] = comp_paymatrix(task.game);

        % attractions
        [env.f_mat_bot,env.f_mat_subj] = deal(NaN(n_trials,n_actions));
        [env.f_mat_bot(1,:),env.f_mat_subj(1,:)] = deal(ones(1,n_actions)/n_actions);

        % adaptive noise structure (WSLS-like)
        [env.success,env.curr_beta] = deal(NaN(n_trials,1));
        [env.noisy_trial,env.noise_breaker,env.tie_breaker,env.circuit_breaker,] = deal(zeros(1,n_trials));
        env.success(2)   = 1/n_actions; % success(2)
        env.curr_beta(2) = 4; % curr_beta(2)

        env.greedy_dist = (env.pi_subj * ones(n_actions,1) + 1e-4  ./ sum(env.pi_subj * ones(n_actions,1) + 1e-4 ))';
        
        env.level_k = repelem(task.bot.levels',n_trials,1);
        
        % data outputs
        %[data.choice_other,data.score_own,data.score_other] = deal(NaN(n_trials,1)); % <- score MUST NOT be initialized
        data.win  = task.game.win;
        data.loss = task.game.loss;
        data.tie  = task.game.tie;
        data.strat_space = task.game.strat_space;
        
        data.trial = repmat([1:task.n_trials]',task.n_blocks,1);
        
    case 'update'
   
        %% ============================= get action ============================= %%
        %
        % (formerly simulate_trial__LKbot)
        % same as LK, but with adaptive noise based on (subject) success rate target
        %

        n_actions = task.game.strat_space;

        if iTrial == 1 % first trials greedy
            
            data.choice_other(iTrial) = randsample(1:n_actions,1,true,env.greedy_dist);

        else

            % prepare for current level
            curr_k = env.level_k(iTrial);
            n_levels = curr_k + 1;
            bot_resp_k = NaN(n_levels,n_actions); 

            % extract relevant trial
            f_bot = env.f_mat_bot(iTrial-1,:)'; % because p(a) is a prediction based the history *up to* the current trial
            f_subj  = env.f_mat_subj(iTrial-1,:)';  
            assert(~any(isnan([f_bot;f_subj])),'NaN attractions.');

            % noise adaptation
            [beta,noisy_trial,success,tie_breaker,rep_bounds] = get_noise_level(iTrial,data,env);
            env.success(iTrial)   = success;
            env.curr_beta(iTrial) = beta;
            env.noisy_trial(iTrial) = noisy_trial;
            env.tie_breaker(iTrial) = tie_breaker;

    %         % compute strategic response <- consistent hierarchy
    %         for k = 0:curr_k
    %             if k == 0 % subject plays egocentric strategy (<- only strategy neglecting payoff table)
    %                 bot_resp_k(1,:) = softmax_fxn(f_self',beta);         
    %             elseif k == 1 % subject BRs to the other's egocentric strategy
    %                 bot_pred = softmax_fxn(f_opp,beta);
    %                 bot_resp_k(2,:) = softmax_fxn((env.pi_bot*bot_pred')',beta);
    %             else % ...subject adds two steps of reasoning to last estimate of same type (i.e. odd vs even)
    %                 bot_pred = softmax_fxn((env.pi_subj*bot_resp_k(k+1-2,:)')',beta);
    %                 bot_resp_k(k+1,:) = softmax_fxn((env.pi_bot*bot_pred')',beta); % used for higher levels 
    %             end
    %         end

            % compute strategic response <- Version from CHASE paper (no initial softmax for k=1)
            for k = 0:curr_k
                if k == 0 % subject plays egocentric strategy (<- only strategy neglecting payoff table)
                    bot_resp_k(1,:) = softmax_fxn(f_bot',beta); 
                elseif k == 1 % subject BRs to the other's egocentric strategy
                    bot_resp_k(2,:) = softmax_fxn((env.pi_bot*f_subj)',beta);
                elseif k == 2 % subject BRs to the other's BR to his own egocentric strategy
                    bot_pred = softmax_fxn((env.pi_subj*f_bot)',beta);
                    bot_resp_k(3,:) = softmax_fxn((env.pi_bot*bot_pred')',beta); % used for higher levels 
                else % ...subject adds two steps of reasoning to last estimate of same type (i.e. odd vs even)
                    bot_pred = softmax_fxn((env.pi_subj*bot_resp_k(k+1-2,:)')',beta);
                    bot_resp_k(k+1,:) = softmax_fxn((env.pi_bot*bot_pred')',beta); % used for higher levels 
                end
            end
            p = bot_resp_k(end,:);

            % adapt for noisy trials 
            if noisy_trial

                % exclude model-conform action
                [~,max_idx] = max(p);
                p(max_idx) = 0;
                p = p/sum(p);

                % ensure that noisy action isn't repeated too often
                n_max = randi([1 2]);
                prior_noisy_trials = sum(env.curr_beta(max(1,end-n_max+1):end) < 1);
                prior_actions_identical = ~any(diff(data.choice_other(max(1,end-n_max+1):end)));
                if prior_noisy_trials >= n_max && prior_actions_identical
                    p(data.choice_other(end)) = 0;
                    p = p/sum(p);
                    env.noise_breaker(iTrial) = 1;
                end       

            end

            % try to break action streaks (i.e. if same action was chosen too many times in a row by either subj or bot)
            if ~isnan(rep_bounds)
                max_rep_action = randi([rep_bounds]);
                if iTrial >= max_rep_action
                    if ~noisy_trial && k~=1 && (~any(diff(data.choice_own(max(1,end-max_rep_action+1):end))) || ~any(diff(data.choice_other(max(1,end-max_rep_action+1):end))))
                        [~,max_idx] = max(p);
                        p(max_idx) = 0;
                        p(setdiff(1:n_actions,max_idx)) = 1;
                        p = p/sum(p);    
                        env.circuit_breaker(iTrial) = 1;
                    end
                end
            end

            % choose action
            data.choice_other(iTrial) = randsample(1:n_actions,1,true,p);   
            
        end

        %% ====================== update attractions ======================== %%

        al = task.bot.params.alpha;

        currTrial = zeros(n_actions,1);
        currTrial(data.choice_own(iTrial)) = 1; %<
        env.f_mat_subj(iTrial,:) = (1-al) * env.f_mat_subj(max(1,iTrial-1),:) + al .* currTrial';

        currTrial = zeros(n_actions,1);
        currTrial(data.choice_other(iTrial)) = 1; %<
        env.f_mat_bot(iTrial,:) = (1-al) * env.f_mat_bot(max(1,iTrial-1),:) + al .* currTrial';

        %% ======================== compute scores ========================== %%
        
        data.score_own(iTrial) = env.pi_subj(data.choice_own(iTrial),data.choice_other(iTrial));
        data.score_other(iTrial) = env.pi_bot(data.choice_other(iTrial),data.choice_own(iTrial));
        
end
    
end

function [beta,noisy_trial,success,tie_breaker,rep_bounds] = get_noise_level(iTrial,data,env)

data.score_own(isnan(data.score_own),:) = [];

level_k = env.level_k(iTrial);

rep_bounds  = NaN; % circuit breaker
success_criterion = 0.5;
time_horizon      = 5;
skeweness         = 1.3;

switch level_k
    case 0, streak_bounds = [1,2];
    otherwise, streak_bounds = [1,3];
end

streak_max = randi([3 4]); % for loose/tie

types = {'wins','ties','losses'};
for i_type = 1:3
    type = types{i_type};
    switch type
        case 'wins',   recent.(type) = (data.score_own(max(1,iTrial-time_horizon):end) > 0); n_trials_back = streak_bounds(2);
        case 'ties',   recent.(type) = (data.score_own(max(1,iTrial-streak_max):end) == 0);  n_trials_back = streak_max;
        case 'losses', recent.(type) = (data.score_own(max(1,iTrial-streak_max):end) < 0);   n_trials_back = streak_max;
    end
    for n_trials = 1:n_trials_back%streak_bounds(2)
        if sum(recent.(type)(max(1,end-n_trials+1):end)) < n_trials
            streak.(type) = n_trials-1;
            break
        else
            streak.(type) = n_trials;
        end
    end
end

success = sum(recent.wins)/numel(recent.wins);

beta = 10;
noisy_trial = 0;
tie_breaker = 0;
if streak.losses >= streak_max || streak.ties >= streak_max
    beta = 1e-3;
    [noisy_trial,tie_breaker] = deal(1);
elseif success >= success_criterion && streak.wins >= streak_bounds(1)
    chance_level = (1/(streak_bounds(2) + 1 - streak.wins))^skeweness;
    if rand < chance_level || streak.wins >= streak_bounds(2)
    	beta = 1e-3;
        noisy_trial = 1;
    end
end

end

function p_actions = softmax_fxn(inputs,beta)

p_actions = exp(beta*inputs)/(sum(exp(beta*inputs)));

end