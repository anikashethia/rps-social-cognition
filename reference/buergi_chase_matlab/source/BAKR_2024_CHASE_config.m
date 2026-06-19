function settings = BAKR_2024_CHASE_config(architecture,maxK_type,maxK,learning_rule)
% 
% Specify all settings/configurations for the model fitting loop
% (see BAKR_2024_CHASE_model for details on the model)
%
% Required specifications:
% - name: string specifying the name of the model
% - loglik_fxn: function handle to the model function which takes data, model
%               configurations (i.e. the output struct from this function) and a
%               set of parameters as input and returns the log-likelihood of the
%               data, given the model
% - params: 1-by-n struct array specifying the n free parameters of the model
%    . name: string specifying the name of the parameter
%    . support: 1x2 double specifying lower and upper bound of the support of the
%              parameter (can be -Inf/Inf)
%    . grid: 1-by-n double specyfing the values for an initial grid search in the
%           parameter space (whos best-fitting values will be used as starting
%           points for an optimization function in the second step)
%    . space: string specifying the estimation space (if support is provided, this
%            will automatically be deduced from that); one of 'integer', 'log',
%            'logit', 'logit_norm'
%

% Niklas Buergi, 2024

%% inputs
if nargin < 4
    warning('Not all inputs provided - using default values.');
    architecture  = 'CH'; % 'CH', 'LK'
    maxK_type     = 'fitted'; % 'fixed', 'fitted' (= can also be lower than exp_max_k)
    maxK          = 3;
    learning_rule = 'RW-freq';
end

%% model specification
S.name       = sprintf('CHASE_%s_k-%s_max-%i_%s',architecture,maxK_type,maxK,learning_rule);
S.loglik_fxn = @BAKR_2024_CHASE_model;
S.sim_fxn    = @BAKR_2024_CHASE_model;

S.architecture  = architecture;
S.learning_rule = learning_rule;
S.maxK_type     = maxK_type;
S.exp_max_k     = maxK;

%% fitting

n_samples = 5;

params(1).name = 'beta';
% params(1).support = [0,10];
params(1).support = [0,100];
params(1).grid = linspace(0.1,5,n_samples);
    
if strcmp(architecture,'CH')    

    params(end+1).name = 'lambda'; % formerly lossav
    % params(end).support = [0,10];
    params(end).support = [0,100];
    % params(end).grid = linspace(1e-4,1.5,n_samples);
    params(end).grid = linspace(1e-4,2,n_samples);

    params(end+1).name  = 'gamma'; % formerly rho
    params(end).support = [0,10];
    % params(end).grid = linspace(0.1,2,n_samples);
    params(end).grid = linspace(1e-4,2,n_samples);
    
end

switch learning_rule
        
    case {'RW-freq','RW-reward'}
        
        params(end+1).name  = 'alpha';
        params(end).support = [0,1];
        % params(end).grid = linspace(0.3,0.99,n_samples);
        params(end).grid = linspace(1e-3,0.99,n_samples);

    case {'RW-hybrid','RW-regret'}
        
        params(end+1).name  = 'alpha';
        params(end).support = [0,1];
        
        params(end+1).name  = 'delta';
        params(end).support = [0,1];

    case 'EWA-full'
        
        params(end+1).name  = 'rhoEWA';
        params(end).support = [0,1];
        
        params(end+1).name  = 'phi';
        params(end).support = [0,1];
        
        params(end+1).name  = 'delta';
        params(end).support = [0,1];
        
    case 'EWA-single' 
        
        % no parameter (except beta)
        
    otherwise
        
        error('Unknown learning rule.');
        
end
     
if strcmp(maxK_type,'fitted')
    params(end+1).name  = 'kappa'; % maxK
    params(end).support = [0,maxK];
    params(end).space = 'integer';
end

params = mn_createParamTable(params,n_samples);
params = table2struct(params)';

%% out
S.params = params;
settings = S;

end
