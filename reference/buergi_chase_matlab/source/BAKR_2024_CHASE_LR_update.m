function [f_mat_own,f_mat_other,varargout] = BAKR_2024_CHASE_LR_update(LR_type,iTrial,f_mat_own,f_mat_other,choice_own,choice_other,params,varargin)
% 
% update the level-0 attractions, based on a specified learning rule and a given set of parameters
%
% update is based on the actions of the *current trial* and thus gives the attractions that the 
% agent will use on the *next* trial. however, as the update presumably happens during feedback
% of the current trial, this is where the corresponding quantities are stored.
%

nActions = size(f_mat_own,2);
past_trial = max(1,iTrial-1);
curr_trial = iTrial;

switch LR_type

    case 'RW-freq' % Rescorla-Wagner learning of frequency (parameter determines learning rate)
        
        al = params.alpha;
        
        currTrial = zeros(nActions,1);
        currTrial(choice_own(curr_trial)) = 1; %<
        f_mat_own(curr_trial,:) = (1-al) * f_mat_own(max(1,past_trial),:) + al .* currTrial';

        currTrial = zeros(nActions,1);
        currTrial(choice_other(curr_trial)) = 1; %<
        f_mat_other(curr_trial,:) = (1-al) * f_mat_other(max(1,past_trial),:) + al .* currTrial';
        
    case 'RW-reward' % Rescorla-Wagner learning of reward (parameter determines learning rate)
        
        al = params.alpha;

        currTrial = zeros(nActions,1);
        currTrial(choice_own(curr_trial)) = varargin{1}(curr_trial); %< score_own
        f_mat_own(curr_trial,:) = (1-al) * f_mat_own(past_trial,:) + al .* currTrial';

        currTrial = zeros(nActions,1);
        currTrial(choice_other(curr_trial)) = varargin{2}(curr_trial); %< score_other
        f_mat_other(curr_trial,:) = (1-al) * f_mat_other(past_trial,:) + al .* currTrial';
        
    case 'RW-hybrid' % Rescorla-Wagner HYBRID learner
        
        al = params.alpha;
        delta = params.delta;

        currTrial = deal(zeros(nActions,1));
        currTrial(choice_own(curr_trial)) = delta*varargin{1}(curr_trial) + (1-delta)*1; % weighted combo of reward and chosen action
        f_mat_own(curr_trial,:) = (1-al) * f_mat_own(past_trial,:) + al .* currTrial';

        currTrial = zeros(nActions,1);
        currTrial(choice_other(curr_trial)) = varargin{2}(curr_trial); %< score_other    <<<<<<< WHY NO HYBRID HERE??
        f_mat_other(curr_trial,:) = (1-al) * f_mat_other(past_trial,:) + al .* currTrial';

    case 'RW-regret' % EWA-style extension of RW (parameters determine learning rate for chosen and unchosen actions, respectively)

        al = params.alpha;
        ga = params.delta;
        
        idxChosen   = choice_own(curr_trial);
        idxUnchosen = 1:nActions;
        idxUnchosen(idxChosen) = [];
        pi_potential = varargin{1}(curr_trial,:); % score_potential_own
        f_mat_own(curr_trial,idxChosen)   = (1-al) * f_mat_own(past_trial,idxChosen)   + al .* pi_potential(idxChosen); 
        f_mat_own(curr_trial,idxUnchosen) = (1-al) * f_mat_own(past_trial,idxUnchosen) + al*ga .* pi_potential(idxUnchosen); % same decay, but different responsivity to reward
        
        idxChosen   = choice_other(curr_trial);
        idxUnchosen = 1:nActions;
        idxUnchosen(idxChosen) = [];
        pi_potential = varargin{2}(curr_trial,:); % score_potential_other
        f_mat_other(curr_trial,idxChosen)   = (1-al) * f_mat_other(past_trial,idxChosen)   + al .* pi_potential(idxChosen); 
        f_mat_other(curr_trial,idxUnchosen) = (1-al) * f_mat_other(past_trial,idxUnchosen) + al*ga .* pi_potential(idxUnchosen); 

    case 'EWA-full' % except the initial values, those are set

        % A(0) = uniform; (initial attraction)
        % N(0) = 0; (prior for belief vs RL models; as in 1-param version below)
        
        rho = params.rhoEWA;  % forgetting
        phi = params.phi;     % discounting
        delta = params.delta; % learning from foregone payoffs

        N = varargin{3};
        
        N(curr_trial)   = rho*N(past_trial) + 1;
        
        idxChosen   = choice_own(curr_trial);
        idxUnchosen = 1:nActions;
        idxUnchosen(idxChosen) = [];
        pi_potential = varargin{1}(curr_trial,:); % score_potential_own
        f_mat_own(curr_trial,idxChosen)   = (phi*N(past_trial) * f_mat_own(past_trial) +   1   * pi_potential(idxChosen)) / N(curr_trial);
        f_mat_own(curr_trial,idxUnchosen) = (phi*N(past_trial) * f_mat_own(past_trial) + delta * pi_potential(idxUnchosen)) / N(curr_trial);
         
        idxChosen   = choice_other(curr_trial);
        idxUnchosen = 1:nActions;
        idxUnchosen(idxChosen) = [];
        pi_potential = varargin{2}(curr_trial,:); % score_potential_other
        f_mat_other(curr_trial,idxChosen)   = (phi*N(past_trial) * f_mat_other(past_trial) +   1   * pi_potential(idxChosen)) / N(curr_trial);
        f_mat_other(curr_trial,idxUnchosen) = (phi*N(past_trial) * f_mat_other(past_trial) + delta * pi_potential(idxUnchosen)) / N(curr_trial);

        [varargout{1},varargout{2}] = deal(N);
    
    case 'EWA-single' % i.e. self-tuning, one param (only beta)

        % rho (forgetting), phi (discounting), delta (weight on forgone)
        % A(0) (initial attraction), N(0) (prior for belief vs RL models)
        
        % one parameter version:
        % - N(0) = 0
        % - A(0) = set by CH model with empirically derived parameter
        % - kappa = 0 (formerly rho; averaging rather than cumulative RL)
        % - phi(t) = change detector function
        % - delta(t) = attention function
        
        N_own = varargin{3};
        N_other = varargin{4};
        
        % 1) change-detector function
        choice_mat = mn_idx2mat(choice_other(1:curr_trial),nActions,'remove_nan');
        history = sum(choice_mat)/size(choice_mat,1);
        current = mn_idx2mat(choice_other(curr_trial),nActions,'remove_nan');
        surprisal = sum((history - current).^2);
        phi = 1 - (1/2)*surprisal;
        
        % 2) learning of foregone payoffs (only if >= received payoff)
        idxChosen   = choice_own(curr_trial);
        pi_potential = varargin{1}(curr_trial,:); % score_potential_own
        delta_vec = double(pi_potential >= pi_potential(idxChosen));
        
        % 3) update
        N_own(curr_trial) = N_own(past_trial) * phi + 1;
        f_mat_own(curr_trial,:) = (phi*N_own(past_trial) * f_mat_own(past_trial) + delta_vec .* pi_potential(idxChosen)) / N_own(curr_trial);
         
        % 1-3 all for other
        choice_mat = mn_idx2mat(choice_own(1:curr_trial),nActions,'remove_nan');
        history = sum(choice_mat)/size(choice_mat,1);
        current = mn_idx2mat(choice_own(curr_trial),nActions,'remove_nan');
        surprisal = sum((history - current).^2);
        phi = 1 - (1/2)*surprisal;
        
        idxChosen   = choice_other(curr_trial);
        pi_potential = varargin{2}(curr_trial,:); % score_potential_other
        delta_vec = double(pi_potential >= pi_potential(idxChosen));
        
        N_other(curr_trial) = N_other(past_trial) * phi + 1;
        f_mat_other(curr_trial,:) = (phi*N_other(past_trial) * f_mat_other(past_trial) + delta_vec .* pi_potential(idxChosen)) / N_other(curr_trial);

        varargout{1} = N_own;
        varargout{2} = N_other;
        
end

end