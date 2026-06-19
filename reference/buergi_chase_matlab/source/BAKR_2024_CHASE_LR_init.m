function [f_mat_own,f_mat_other,varargout] = BAKR_2024_CHASE_LR_init(type,nTrials,nActions,varargin)

switch type
    
    case 'RW-freq' % assume uniform prior
        [f_mat_own,f_mat_other] = deal(NaN(nTrials,nActions));
        [f_mat_own(1,:),f_mat_other(1,:)] = deal(ones(1,nActions)/nActions);
        
    case {'RW-reward','RW-hybrid'} % assume prior = expected value for uniform opponent 
        pi_own = varargin{1};
        pi_other = varargin{2};
        [f_mat_own,f_mat_other] = deal(NaN(nTrials,nActions));
        f_mat_own(1:2,:)   = repmat(mean(pi_own,2)',[2,1]) / 4; % <- should be by n_actions?!
        f_mat_other(1:2,:) = repmat(mean(pi_other,2)',[2,1]) / 4;
        
    case {'EWA-full','EWA-single','RW-regret'}
        pi_own   = varargin{1};
        pi_other = varargin{2};  
        [f_mat_own,f_mat_other] = deal(NaN(nTrials,nActions));
        f_mat_own(1:2,:)   = repmat(mean(pi_own,2)',[2,1]); % expected pi ignoring opp
        f_mat_other(1:2,:) = repmat(mean(pi_other,2)',[2,1]);
        
        % compute potential payoffs (for all possible actions) 
        choice_own   = varargin{3};
        choice_other = varargin{4};
        choice_mat_other = mn_idx2mat(choice_other,nActions,'remove_nan');
        varargout{1} = (pi_own * choice_mat_other')'; % score_pot_own
        choice_mat_own = mn_idx2mat(choice_own,nActions,'remove_nan');
        varargout{2} = (pi_other * choice_mat_own')'; % score_pot_other   
        
        if any(strcmp(type,{'EWA-full','EWA-single'}))
            varargout{3} = [0; NaN(size(choice_other,1)-1,1)]; % N ("observation equivalents")
            varargout{4} = [0; NaN(size(choice_other,1)-1,1)]; % 
        end

end

end