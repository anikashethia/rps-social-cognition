function [negLL,states,data] = BAKR_2024_CHASE_model(data,model,curr_params,type)
%
% Code to fit the CHASE model (Buergi, Aydogan, Konovalov, & Ruff; under review)
%
% In brief, this model captures mentalization in strategic interactions by assuming that:
% - Non-strategic players (defined as k = 0) have a tendency to repeat their historical action frequencies.
% - Strategic players try to exploit this tendency by performing a limited number of recursive reasoning steps 
%   (k > 0), i.e. iteratively (and noisily) best-responding to that action distribution.
% - Finally, adaptive players (kappa >= 2) assume that they are facing a strategic player and try to infer 
%   their level of recursive reasoning, by integrating evidence for the different levels over time.
% 
% Inputs
%   - data: struct containing the experimental data (i.e. choices in Rock-Paper-Scissors) of both players
%      . choice_own: vector containing the chosen actions of the subject (from 1:3)
%      . choice_other: vector containing the chosen actions of the opponent (from 1:3)
%   - curr_params: vector containing the model parameters
%      . alpha: speed of updating attractions (i.e. learning rate of a delta rule over chosen actions)
%      . beta: recursive reasoning noise (i.e. softmax inverse decision temperature)
%      . gamma: sensitivity to evidence for the sophistication of the opponent
%      . lambda: loss aversion (or lack thereof)
%      . kappa: depth of mentalization (note: equal to k if kappa < 2)
% Outputs
%   - negLL: scalar negative log-likelihood of the data, given the model and the parameters
%   - states: struct containing the inferred internal states (for e.g. use in neuroimaging analysis)
%      . SV: subjective value of the chosen action, given the current predictions about the opponent
%      . APE: opponent action prediction error, i.e. level of surprise about the observed opponent action
%      . BU: opponent sophistication belief update, i.e. the Kullback-Leibler divergence between successive
%        beliefs
%

% Niklas Buergi, 2024

%% Initialize

switch type
    
    case 'fit'

        % set game defaults, if none provided
        if ~any(isfield(data,{'strat_space','win','loss','tie'}))
            data.win = 1;
            data.loss = -1;
            data.tie = 0;
            data.strat_space = 3;
        end
        
        % extract variables
        choice_own   = data.choice_own;
        choice_other = data.choice_other;
        % nTrials  = size(choice_own,1);
        
        assert(numel(choice_own) > 1 & numel(choice_other) > 1 & numel(data.missing) > 1);

        
    case 'sim'
        
        task = data;
        
        data = struct();
        [data,env] = task.fxn(data,task,'init');
        data.n_trials = numel(data.trial);
        data.missing = zeros(data.n_trials,1);
        
        [data.score_own,data.score_other,choice_own,choice_other] = deal(NaN(data.n_trials,1));
        
end
        
nTrials = data.n_trials;
nActions = data.strat_space(1);

% name parameters
params = array2table(curr_params','VariableNames',{model.params.name});

% ======================== adapt to parameterization ========================= %

% add loss aversion, if parameter provided
if any(strcmp(params.Properties.VariableNames,'lambda')) % lossav
    data.loss = data.loss*params.lambda;
end

% construct payoff matrices
[pi_own, pi_other] = comp_paymatrix(data);

% set exp_max_k
if strcmp(model.maxK_type,'fitted') % || strcmp(type,'sim')
    exp_max_k = params.kappa;
else
    exp_max_k = model.exp_max_k; % i.e. if not provided via parameters, take from settings
end

% ========================= pre-allocate variables =========================== %

% main variables
subj_pred = NaN(nTrials,nActions); % subject's estimated probability of the different possible opponent actions
subj_resp = NaN(nTrials,nActions); % subject's response probabilities to the estimate above (based on a noisy best-response)

subj_KL_div = NaN(nTrials,1); % belief update (BU), given by the KL-divergence between successive beliefs
subj_APE = NaN(nTrials,1); % 1 minus the probability assigned to the action chosen by the opponent (from subj_pred)
subj_SV = NaN(nTrials,1); % subjective value of choosing the different actions, given one's predictions of the opponent

% recursive reasoning steps (either from own perspetive ('LK'), or from the perspective of the opponent ('CH-leaky'))
if exp_max_k < 2 || strcmp(model.architecture,'LK')
    subj_pred_k = deal(NaN(exp_max_k+1,nActions)); % 1st order beliefs (what subjects think the opponent will play)
    subj_resp_k = deal(NaN(exp_max_k+1,nActions)); % subject's response probabilities (BR), in response to their beliefs above
elseif exp_max_k >= 2 && contains(model.architecture,'CH')
    opp_pred_oppk = NaN(exp_max_k,nActions); % 2nd order beliefs (what subject thinks opponent thinks about them)
    opp_resp_oppk = NaN(exp_max_k,nActions); % opponent's response probabilities, in response to their beliefs above (as perceived by the subject)
    subj_beliefs = ones(1,exp_max_k)/exp_max_k; % priors of the subject about the opponents's current level k, bounded by their own sophistication
end

% outputs for figures
beliefs = NaN(nTrials,3);
likelihoods = NaN(nTrials,3);

% initialize attractions
switch model.learning_rule
    case 'RW-freq'
        [f_mat_own,f_mat_other] = BAKR_2024_CHASE_LR_init(model.learning_rule,nTrials,nActions); 
    case {'RW-reward','RW-hybrid'}
        [f_mat_own,f_mat_other] = BAKR_2024_CHASE_LR_init(model.learning_rule,nTrials,nActions,pi_own,pi_other);
        score_own   = data.score_own; score_other = data.score_other;
    case 'RW-regret'
        [f_mat_own,f_mat_other,score_pot_own,score_pot_other] = BAKR_2024_CHASE_LR_init(...
                                                      model.learning_rule,nTrials,nActions,...
                                                      pi_own,pi_other,...
                                                      choice_own,choice_other);
    case {'EWA-full','EWA-single'}
        [f_mat_own,f_mat_other,score_pot_own,score_pot_other,N_own,N_other] = BAKR_2024_CHASE_LR_init(...
                                                      model.learning_rule,nTrials,nActions,...
                                                      pi_own,pi_other,...
                                                      choice_own,choice_other);
end

% =========================== save initial values ============================ %

f_mat_own_0 = f_mat_own(1,:);
f_mat_other_0 = f_mat_other(1,:);
if exp_max_k >= 2 && contains(model.architecture,'CH')
    subj_beliefs_0 = subj_beliefs;
end
if any(strcmp(model.learning_rule,{'EWA-full','EWA-single'}))
    N_own_0 = N_own(1);
    N_other_0 = N_other(1);
end

%% Trial loop
for iTrial = 1:nTrials
    
%     if any(data.choice_own == 0 | isnan(data.choice_own))
%         error('Missing choices - have to adapt first.');
%     end

    % skip updates if missing trial
    if numel(data.missing) > 1 && data.missing(iTrial)
        f_mat_own(iTrial,:) = f_mat_own(max(1,iTrial-1),:);
        f_mat_other(iTrial,:) = f_mat_other(max(1,iTrial-1),:);
        if any(strcmp(model.learning_rule,{'EWA-full','EWA-single'}))
            N_own(iTrial,:) = N_own(max(1,iTrial-1),:);
            N_other(iTrial,:) = N_other(max(1,iTrial-1),:);
        end
        continue
    end
    
    % re-set initial values (if fitting across blocks)
    if data.trial(iTrial) == 1 && iTrial > 1
        f_mat_own_old   = f_mat_own(iTrial-1,:);
        f_mat_other_old = f_mat_other(iTrial-1,:);
        f_mat_own(iTrial-1,:)   = f_mat_own_0; % need to overwrite old ones to ensure correct update below
        f_mat_other(iTrial-1,:) = f_mat_other_0;
        if exp_max_k >= 2 && contains(model.architecture,'CH')
            subj_beliefs = subj_beliefs_0; % will be overwritten at end of trial
        end
        if any(strcmp(model.learning_rule,{'EWA-full','EWA-single'}))
            N_own_old   = N_own(iTrial-1);
            N_other_old = N_other(iTrial-1);
            N_own(iTrial-1) = N_own_0;
            N_other(iTrial-1) = N_other_0;
        end
    end   
    
    % strategic reasoning
    if exp_max_k < 2 || strcmp(model.architecture,'LK')
                    
        % ======================================================================
        %                       Low-level or static reasoning
        % ======================================================================

        for k = 0:exp_max_k
            
            % ========================= (Re-)active strategies =================== %

            if k == 0 % egocentric strategy (ignoring opponent)
                subj_pred_k(1,:) = ones(1,nActions)/nActions; 
                subj_resp_k(1,:) = softmax_fxn(f_mat_own(max(1,iTrial-1),:),params.beta); 

            elseif k == 1 % responding to the other's egocentric strategy
                subj_pred_k(2,:) = softmax_fxn(f_mat_other(max(1,iTrial-1),:),params.beta);
                subj_resp_k(2,:) = softmax_fxn((pi_own*subj_pred_k(2,:)')',params.beta);

            elseif k >= 2

                % ===================== LK: Higher-level reasoning =================== %

                % always add two steps of reasoning to last estimate of same type (i.e. odd vs even)
                subj_pred_k(k+1,:) = softmax_fxn((pi_other*subj_resp_k(k+1-2,:)')',params.beta);
                subj_resp_k(k+1,:) = softmax_fxn((pi_own*subj_pred_k(k+1,:)')',params.beta); 

            end
            
            subj_pred(iTrial,:) = subj_pred_k(end,:);
            subj_resp(iTrial,:) = subj_resp_k(end,:);
            
        end

    elseif exp_max_k >= 2 && contains(model.architecture,'CH')
                    
        % ======================================================================
        %                     CH: Belief-based strategies
        % ======================================================================

        % subject simulates reasoning process *of the opponent* to:
        % - form and update beliefs about the opponent's level of reasoning
        % - predict the next opponent action, marginalizing over these beliefs
        
        for k = 2:exp_max_k

            if k == 2 % opponent levels 0 & 1
                opp_pred_oppk(1,:) = ones(1,nActions)/nActions; % i.e. no expectations; used only for computing subject action predictions
                opp_resp_oppk(1,:)  = softmax_fxn(f_mat_other(max(1,iTrial-1),:),params.beta); % = exp_p_a_opp(2,:)

                opp_pred_oppk(2,:) = softmax_fxn(f_mat_own(max(1,iTrial-1),:),params.beta); % = exp_p_a_subj(1,:)
                opp_resp_oppk(2,:)  = softmax_fxn((pi_other*opp_pred_oppk(2,:)')',params.beta);

            elseif k > 2 % higher levels: add two more steps to last action estimate of same type (i.e. odd vs even)
                opp_pred_oppk(k,:) = softmax_fxn((pi_own*opp_resp_oppk(k-2,:)')',params.beta); % ..compute their opponent's best response to that..
                opp_resp_oppk(k,:)  = softmax_fxn((pi_other*opp_pred_oppk(k,:)')',params.beta); % ..and compute their own best response to that

            end
            
        end

        % ========================= Subject beliefs ========================== %

        % subject's weighted opponent action prediction & response to that
        subj_pred(iTrial,:) = subj_beliefs * opp_resp_oppk;
        subj_resp(iTrial,:) = softmax_fxn(pi_own*subj_pred(iTrial,:)',params.beta);
        
    end
    
    % =========================== Action simulation ========================== %
    
    if strcmp(type,'sim')
        
        % subject
        choice_own(iTrial) = find(mnrnd(1,subj_resp(iTrial,:)));
        data.choice_own(iTrial) = choice_own(iTrial);

        % bot
        [data,env] = task.fxn(data,task,'update',env,iTrial); % need only: choice_other(iTrial)
        choice_other(iTrial,1) = data.choice_other(iTrial);
        score_own(iTrial,1)    = data.score_own(iTrial);
        score_other(iTrial,1)  = data.score_other(iTrial);

    end
    
    % ========================= Subject beliefs ========================== %
    
    if exp_max_k >= 2 && contains(model.architecture,'CH')

        % belief update
        prior = subj_beliefs;
        likelihood = softmax_fxn(opp_resp_oppk(:,choice_other(iTrial)),params.gamma);
        posterior = (likelihood .* prior') ./ sum(likelihood .* prior');
        subj_KL_div(iTrial) = nansum(posterior .* log(posterior ./ prior'));

        % save
        beliefs(iTrial,1:exp_max_k) = subj_beliefs;
        likelihoods(iTrial,1:exp_max_k) = likelihood;
        
        % replace with new belief
        subj_beliefs = posterior';
        
    end
    
    % compute internal states
    subj_APE(iTrial) = 1 - subj_pred(iTrial,choice_other(iTrial));
    subj_SV(iTrial) = pi_own(choice_own(iTrial),:) * subj_pred(iTrial,:)';
    
    % ==========================================================================
    %                             Level-0 update
    % ==========================================================================
    
    % L0 representation
    switch model.learning_rule
        case 'RW-freq'
            [f_mat_own,f_mat_other] = BAKR_2024_CHASE_LR_update(model.learning_rule,iTrial,f_mat_own,f_mat_other,...
                                                        choice_own,choice_other,params);
        case {'RW-reward','RW-hybrid'}
            [f_mat_own,f_mat_other] = BAKR_2024_CHASE_LR_update(model.learning_rule,iTrial,f_mat_own,f_mat_other,...
                                                        choice_own,choice_other,params,...
                                                        score_own,score_other);
        case 'RW-regret'
            [f_mat_own,f_mat_other] = BAKR_2024_CHASE_LR_update(model.learning_rule,iTrial,f_mat_own,f_mat_other,...
                                                        choice_own,choice_other,params,...
                                                        score_pot_own,score_pot_other);
        case {'EWA-full','EWA-single'}
            [f_mat_own,f_mat_other,N_own,N_other] = BAKR_2024_CHASE_LR_update(model.learning_rule,iTrial,f_mat_own,f_mat_other,...
                                                        choice_own,choice_other,params,...
                                                        score_pot_own,score_pot_other,N_own,N_other);
    end

    % put overwritten values back (if fitting across blocks) <- cannot do earlier because this is used for update above
    if data.trial(iTrial) == 1 && iTrial > 1
        f_mat_own(iTrial-1,:) = f_mat_own_old;
        f_mat_other(iTrial-1,:) = f_mat_other_old;
        if any(strcmp(model.learning_rule,{'EWA-full','EWA-single'}))
            N_own(iTrial-1) = N_own_old;
            N_other(iTrial-1) = N_other_old;
        end
    end

end

%% Post-processing

% internal states
states.subj_SV = subj_SV;
states.subj_APE = subj_APE;
states.subj_KL_div = subj_KL_div;

% states for figures
% states.f_mat_own  = f_mat_own;
% states.likelihoods = likelihoods;
states.beliefs = beliefs;

switch type
    
    case 'fit'

        lik = NaN(nTrials,1);
        lik(~data.missing) = subj_resp(sub2ind([nTrials,nActions],find(~data.missing),choice_own(~data.missing))); % chosen action
        states.lik = lik;
        % states.p_a = subj_resp;
        
        negLL = -sum(log(lik(~data.missing)));

    
    case 'sim'

        states.beliefs = beliefs;
        states.level_k = env.level_k;
        % states.score_own = score_own;

        states.score = NaN(nTrials,1);
        for i_block = 1:task.n_blocks
            idx = (task.n_trials*(i_block-1))+1:(task.n_trials*(i_block));
            states.score(idx) = cumsum(score_own(idx));
        end
        % states.gamma = curr_gamma(1:nTrials);

        data.choice_own = choice_own;
        data.choice_other = choice_other;
        data.score_own = score_own;
        data.score_other = score_other;

        data.n_blocks = 1;

        data.bot_level = task.bot.levels;
        
        negLL = NaN;
        
end

end

function p_actions = softmax_fxn(inputs,beta)

p_actions = exp(beta*inputs)/(sum(exp(beta*inputs)));

end
