function MN = mn_fit(dataset,models,config)

fprintf('\n+++ Fitting +++\n\n\n');

% check input format
if numel(models) > 1
    assert(isa(models,'cell'),'If providing multiple models, please use cell arrays.');
end

% get merlin settings, if not provided
if nargin < 3
    config  = mn_fit_config();
end

% determine number of models to fit
nModels = numel(models);
if isa(models,'struct')
    models = {models};
end

% loop over models
for iModel = 1:nModels
    
    fprintf('\bModel %i/%i: ',iModel,nModels);
    model = models{iModel};
    
    % if doing MAP, set support and estimation space (or adjust, if needed)
    if strcmp(config.objective,'MAP')

        % check if all provided
        params = struct2table(model.params);
        assert(isfield(model.params,'prior'),'If setting the optimization objective to MAP, need to specify a prior distribution for each parameter.');
        assert(~any(cellfun(@isempty,params.prior),'all'),'Please provide prior either for all or for no parameter.');
        if any(strcmp(fieldnames(model.params),'support'))
            provided_support = params.support;
        end

        % define support and estimation space based on prior distributions (only if not integer)
        idx_non_integer = find(~strcmp(params.space,'integer'))';
        for i_param = idx_non_integer%1:height(params)
            switch params.prior{i_param,1}
                case 'normal'
                    params.support(i_param,:) = [-Inf,Inf];
                    params.space{i_param} = 'linear';
                case 'gamma'
                    params.support(i_param,:) = [0,Inf];
                    params.space{i_param} = 'log';
                case 'beta'
                    params.support(i_param,:) = [0,1];
                    params.space{i_param} = 'logit';
                case 'uniform'
                    params.support(i_param,:) = [params.prior{i_param,2:3}];
                    params.space{i_param} = 'logit_norm';
                otherwise
                    error('Unknown prior distribution.');
            end
        end
        model.params = table2struct(params)';

        % check if any were changed, and give warning if yes
        if exist('provided_support','var')
            idx_diff = find(~all(params.support == provided_support,2));
            if ~isempty(idx_diff)
                warning('Parameter support was adjusted to match prior distributions (for: %s).',strjoin(params.name(idx_diff),', '));
            end
        end

    end

    % check that grid is never outside support
    all_inside = ones(numel(model.params),1);
    for i_param = 1:numel(model.params)
        support = model.params(i_param).support;
        if numel(support) > 1
            grid = model.params(i_param).grid;
            all_inside(i_param) = all(grid >= support(1) & grid <= support(2));
        end
    end
    assert(all(all_inside),'Grid values outside support for %i parameter(s): %s.',sum(~all_inside),strjoin({model.params(~all_inside).name},', '));

    % fit
    MN(iModel) = mn_subjectLoop(dataset,model,config);
    
end

fprintf('\bAll models completed.\n\n');

end