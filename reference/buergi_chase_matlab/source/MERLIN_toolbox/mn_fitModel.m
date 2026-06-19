function [params,states,optim] = mn_fitModel(data,model,param_combs,config)
%
% add manual input of starting points as alternativ option
%

%% prep
if nargin < 4
    config = mn_fit_config();
end
nParams = size(model.params,2);
states_all = [];

% define objective function
switch config.objective
    case 'ML'
        loss_fxn = model.loglik_fxn;
    case 'MAP'
        loss_fxn = @mn_LL2AP;
end

% fit parameters (if there are any)
if nParams > 0
    
    %% grid search
    if config.use_grid && strcmp(config.type_init,'grid')
        
        % create an n-d grid and vectorize it (with n = nParams)
        if nargin < 3 || isempty(param_combs)
            [param_combs,n_combs] = mn_getGridMat(model.params.grid);
        else
            n_combs = size(param_combs,1);
        end
        
    else
        
        % define optimization starting points randomly
        n_combs = config.optim_n_it;   
        switch config.type_init
            case 'prior',   sampling_fxn = @mn_samplePrior;
            case 'uniform', sampling_fxn = @mn_sampleUniform;
        end  
        param_combs = NaN(n_combs,nParams);
        for iParam = 1:nParams     
            param_combs(:,iParam) = sampling_fxn(model,iParam,n_combs);
            if strcmp(model.params(iParam).space,'integer') % snap integer parameters to nearest value
                p_valid = min(model.params(iParam).grid):max(model.params(iParam).grid);
                param_combs(:,iParam) = p_valid(dsearchn(p_valid',param_combs(:,iParam)));
            end
        end 

        % if integer parameters, repeat random starting points for each value of it
        idx_int = find(strcmp({model.params.space},'integer'));
        if any(idx_int)
            int_vals = model.params(idx_int).grid;
            param_combs = repmat(param_combs,numel(int_vals),1);
            param_combs(:,idx_int) = repelem(int_vals',n_combs,1);
            n_combs = size(param_combs,1);
        end
        
    end
    
    % get loss per parameter combination
    loss_per_comb = NaN(n_combs,1);
    for iComb = 1:n_combs
        curr_params = param_combs(iComb,:);
        if config.save_grid
            [loss_per_comb(iComb),states_all{iComb,1}] = loss_fxn(data,model,curr_params','fit'); %,false); % the latter for plotting
        else
            loss_per_comb(iComb) = loss_fxn(data,model,curr_params','fit'); %,false);
        end
    end
        
    % sort parameter combinations by loss % <- now done below (i.e. also for sampling)
    fxn_evals = [param_combs loss_per_comb];
    [evals_sorted,idx_sort] = sortrows(fxn_evals,nParams+1,'ascend'); % careful: *loss* (e.g. *negative* LL)

    loss_best = evals_sorted(1,end);
    p_best = evals_sorted(1,1:end-1);

    %% optimization algorithm
    if config.use_optim

        % identify free (& non-integer) parameters
        idx_free = NaN(nParams,1);
        for iParam = 1:nParams
            idx_free(iParam) = ~(strcmp(model.params(iParam).space,'integer') | numel(model.params(iParam).support) == 1); % i.e. free if point support
        end
        
        % identify integer parameter and do optimization on best grid values for *each* value of the int param
        int_vals = NaN;
        idx_int = find(strcmp({model.params.space},'integer'));
        if any(idx_int)% && strcmp(S.params(idx_int,1),'maxK') % for now only maxK
            assert(numel(idx_int) == 1,'Fitting currently only implemented for one integer parameter.');
            % int_vals = model.params(idx_int).grid;
            int_vals = sort(model.params(idx_int).grid,'descend'); % doesn't change estimates
        end
        
        for curr_int = int_vals % i.e. only single iteration if no int
            
            if any(idx_int) && config.use_grid
                curr_evals = evals_sorted(evals_sorted(:,idx_int) == curr_int,:); % if int, run optimizer for the best LLs for each value of the int
            else
                curr_evals = evals_sorted;
            end
        
            % loop over best combinations from grid search as optimization starting points
            for iRun = 1:min(config.optim_n_it,size(curr_evals,1))

                % only optimize non-integer ones
                curr_p       = curr_evals(iRun,1:end-1);
                curr_p_fixed = curr_p(~idx_free);
                curr_p_free  = curr_p(logical(idx_free));

                f_optim = @(curr_params)loss_fxn(data,model,curr_params,'fit');%,false); 
                f_optim_con = @(curr_p_free)f_optim(assemble_param_vect(curr_p_fixed,curr_p_free,idx_free)); % constrained to free parameters
                f_optim_est = @(curr_p_free)f_optim_con(transform_to_native_space(curr_p_free,idx_free,model)); % transformed back to native space

                curr_p_trans = transform_to_est_space(curr_p_free,idx_free,model); % transform initial parameters into estimation space

                options = optimset('Display','off');
                try
                    [p_optim,loss_optim] = fminunc(f_optim_est,curr_p_trans,options); 
                catch e
                    % warning('Error during optimization (%s: %s). Trying once again.',e.identifier,e.message);
                    [p_optim,loss_optim] = fminunc(f_optim_est,curr_p_trans,options); 
                end
                
%                 [p_optim,loss_optim] = fminsearch(f_optim_est,curr_p_trans,options);
                
%                 gs = GlobalSearch;
%                 problem = createOptimProblem('fmincon','x0',curr_p_trans,'objective',f_optim_est);
% 
% %                 % alternative: untransformed, and passing bounds to optimizer instead: <- only fmincon, and ~3x slower!
% %                 lb = arrayfun(@(param) param.support(1),model.params(find(idx_free)));
% %                 ub = arrayfun(@(param) param.support(2),model.params(find(idx_free)));
% %                 problem = createOptimProblem('fmincon','x0',curr_p_free,'objective',f_optim_con,'lb',lb,'ub',ub);
%                 
%                 [p_optim,loss_optim] = run(gs,problem);
                
                % compare to current argmin (maybe save these too?)
                if loss_optim <= loss_best
                    loss_best = loss_optim;
                    p_best = assemble_param_vect(curr_p_fixed,transform_to_native_space(p_optim,idx_free,model),idx_free)';
                end
                
            end
            
        end

    end

    %% output
    % overview
    estimates = p_best;
    if any(strcmp(config.check_param,{'warning','error'}))
        idx = (p_best > config.max_val | p_best < -config.max_val | isnan(p_best));
        if any(idx)
            param_names = {model.params.name};
            msg = sprintf('Unplausible parameter estimates (%s = %s).',strjoin(param_names(idx),', '),num2str(p_best(idx),log10(config.max_val)+2));
            switch config.check_param
                case 'warning', warning(msg);
                case 'error', error(msg);
            end
        end
    end
%     assert(~(any(p_best > 1e3) | any(p_best < -1e3)),'Unplausible parameter estimates.');

    % get all model output with the optimal parameter values
    [~,states] = loss_fxn(data,model,p_best','fit'); %,0); 
    
elseif nParams == 0
    
    % if no-parameter model, just return the loss (therefore also no MAP/loss_fxn)
    [loss_best,states] = model.loglik_fxn(data,model);
%     estimates = loss;
    estimates = [];
    
end

if ~config.save_grid
    states_all{1} = states;
end

params = struct();
for i_param = 1:size(model.params,2)
    params.(model.params(i_param).name) = estimates(i_param);
end

% pack
if nParams > 1 && config.use_grid && strcmp(config.type_init,'grid')
    optim.whole_grid.states = states_all;
    optim.whole_grid.negLL = loss_per_comb;
end
optim.negLL = loss_best;
optim.AIC = 2*loss_best + 2*numel(model.params);
optim.BIC = 2*loss_best + log(data.n_trials-sum(data.missing))*numel(model.params);

% param_combs <- return, but don't add to struct (so is only added once if multiple subjects)

end

%% ------------------------------ helper fxn -------------------------------- %%
function curr_p = assemble_param_vect(curr_p_fixed,curr_p_free,idx_free)

curr_p = NaN(size(idx_free));
curr_p(logical(idx_free)) = curr_p_free;
curr_p(~idx_free) = curr_p_fixed;

end

function curr_p_trans = transform_to_est_space(curr_p_free,idx_free,S)

pos_free = find(idx_free);
for i_param = 1:sum(idx_free)
    switch S.params(pos_free(i_param)).space
        case 'linear', curr_p_trans(i_param) = curr_p_free(i_param);
        case 'log',    curr_p_trans(i_param) = log(curr_p_free(i_param));
        case {'logit','logit_norm'}
            support = S.params(pos_free(i_param)).support;
            curr_p_norm = (curr_p_free(i_param) - support(1)) / diff(support); % normalize to unit range (if support distinct)
            curr_p_trans(i_param) = -log((1/curr_p_norm) - 1);
    end
end

end 

function curr_p_trans = transform_to_native_space(curr_p_free,idx_free,S)

pos_free = find(idx_free);
for i_param = 1:sum(idx_free)
    switch S.params(pos_free(i_param)).space
        case 'linear', curr_p_trans(i_param) = curr_p_free(i_param);
        case 'log',    curr_p_trans(i_param) = exp(curr_p_free(i_param));
        case {'logit','logit_norm'}
            support = S.params(pos_free(i_param)).support;
            curr_p_norm = 1/(1+exp(-curr_p_free(i_param)));
            curr_p_trans(i_param) = (curr_p_norm * diff(support)) + support(1); % scale/shift back (if support not unit range)
    end
end

end 
