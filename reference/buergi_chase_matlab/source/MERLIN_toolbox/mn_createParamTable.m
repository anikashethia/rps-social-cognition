function params = mn_createParamTable(params,n_samples)

if ~isempty(params)
    
    params = struct2table(params);
    existing_specs = params.Properties.VariableNames;
    
    % determine estimation space based on support (to avoid discontinuities in objective function)
    if any(strcmp(existing_specs,'support'))
        assert(~isa(params.support,'cell'),'Please provide support either for all or for no parameter.');
        if ~any(strcmp(existing_specs,'space'))
            params.space = cell(height(params),1);
        end
        for i_param = 1:height(params)
            if isempty(params.space{i_param})
                support = params.support(i_param,:);
                if support(1) == -Inf    && support(2) == Inf, params.space{i_param} = 'linear';
                elseif support(1) == 0   && support(2) == Inf, params.space{i_param} = 'log';
                elseif support(1) == 0   && support(2) == 1,   params.space{i_param} = 'logit';
                elseif support(1) > -Inf && support(2) < Inf,  params.space{i_param} = 'logit_norm'; % includes everything else (by shifting and scaling)
                else, error('Invalid support specified.');
                end
            end
        end
             
    else % if neither prior nor support provided, assume everything is unbounded  
        params.support = repmat([-Inf,Inf],height(params),1);
        params.space = repmat({'linear'},height(params),1);
    end
    
    % add default starting value grid, if not manually provided <<<- NOT FULLY WORKING YET (if incompletely provided)
    if ~any(strcmp(existing_specs,'grid'))
        params.grid = cell(height(params),1);
    end
    for i_param = 1:height(params)
        if isa(params.grid(i_param,:),'cell')
            curr_param = params.grid{i_param}; % <- default
        else
            curr_param = params.grid(i_param,:);
        end
        if all(curr_param == 0)
            switch params.space{i_param}
                case 'integer', params.grid{i_param} = params.support(i_param,1):params.support(i_param,2);
                case 'linear', params.grid{i_param} = linspace(-10,10,n_samples);
                case 'log', params.grid{i_param} = linspace(0+1e-3,10,n_samples); % avoid boundries
                case {'logit','logit_norm'}
                    support = params.support(i_param,:);
                    params.grid{i_param}  = linspace(support(1)+1e-3,support(2)-1e-3,n_samples); % avoid boundries
            end
        end
    end
    
else
    
    params = table();
    
end

end
