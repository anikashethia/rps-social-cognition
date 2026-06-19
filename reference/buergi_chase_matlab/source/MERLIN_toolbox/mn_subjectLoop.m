function MN = mn_subjectLoop(dataset,model,config)
%
% loops over subjects (and blocks) to fit the model via ML
%

% initialize
n_subj  = size(dataset,1);
fits(n_subj) = struct();
% output  = struct();

% IF GRID SEARCH create parameter combination grid for optim starting points
if ~isempty(model.params) % & not all fixed?
    param_combs = mn_getGridMat(model.params.grid);
else
    param_combs = [];
end

error_log = cell(n_subj,1);
if config.use_parfor
    fprintf('Distributing the work...\n\n');
    parfor i_subj = 1:n_subj
        fprintf('\nSubj %i/%i ',i_subj,n_subj);
        try
            [fits(i_subj).params,fits(i_subj).states,fits(i_subj).optim] = fit_curr_subj(dataset(i_subj,:),model,param_combs,config);
        catch e
            error_log{i_subj}.subj = i_subj;
            error_log{i_subj}.identifier = e.identifier;
            error_log{i_subj}.message = e.message;
            % warning('Couldn''t fit subj %i (%s: %s).',i_subj,e.identifier,e.message);
            fprintf('<- aborted due to %s (%s).',e.identifier,e.message);
        end
    end
    % fprintf('\nDone!\n\n');
    fprintf('\n\nDone!\n\n');
else
    for i_subj = 1:n_subj
        mn_printProgress(i_subj,n_subj,''); % 'Fitting '
        % [fits(i_subj).params,fits(i_subj).states,fits(i_subj).optim] = fit_curr_subj(dataset(i_subj,:),model,param_combs,config);
        try
            [fits(i_subj).params,fits(i_subj).states,fits(i_subj).optim] = fit_curr_subj(dataset(i_subj,:),model,param_combs,config);
        catch e
            error_log{i_subj}.subj = i_subj;
            error_log{i_subj}.identifier = e.identifier;
            error_log{i_subj}.message = e.message;
            % warning('Couldn''t fit subj %i (%s: %s).',i_subj,e.identifier,e.message);
        end
    end
    mn_printProgress(i_subj+1,n_subj,'');
end

% pack
MN.type   = 'fit';
MN.model  = model;
MN.model.grid = param_combs;
MN.config = config;
vars_data = fieldnames(dataset);
vars_fits = fieldnames(fits);
for i_subj = 1:size(dataset,1)
    MN.subj(i_subj,1).subjID = dataset(i_subj).subjID;
    for i_var = 1:numel(vars_data)
        MN.subj(i_subj,1).data.(vars_data{i_var}) = dataset(i_subj).(vars_data{i_var});
    end
    for i_var = 1:numel(vars_fits)
        MN.subj(i_subj,1).(vars_fits{i_var}) = fits(i_subj).(vars_fits{i_var});
    end
end

error_log(cellfun(@isempty,error_log)) = [];
if ~isempty(error_log)
    missing_subj = cellfun(@(entry) entry.subj,error_log);
    missing_text = strjoin(string(missing_subj),', ');
    warning('Couldn''t fit %i subject(s): %s.',numel(error_log),missing_text);
    fprintf('\n\n');
    MN.log = struct2table(cell2mat(error_log));
else
    MN.log = [];
end

if config.save_output
    if config.save.timestamp
        timestamp = datestr(now,'yy-mm-dd_HH-MM-SS');
        filename = sprintf('%s_fits_%s.mat',model.name,timestamp);
    else
        filename = sprintf('%s_fits.mat',model.name);
    end
    if ~isfolder(config.output_folder)
        mkdir(config.output_folder);
    end
    try
        save(fullfile(config.output_folder,filename),'MN');
    catch e
        warning('Couldn''t save fits (%s - %s)',e.identifier,e.message);
    end
end

end

% ---------------------------------------------------------------------------- %

function [params,states_best,optim] = fit_curr_subj(subj_data,model,param_combs,config)

if subj_data.n_blocks > 1 %config.fit_per_block
    n_blocks = subj_data.n_blocks;
    for i_block = 1:n_blocks
        block_data = extract_block_data(subj_data,i_block);        
        [params(i_block),states_best(i_block),states_all(i_block)] = fit_data(block_data,model,param_combs,config); % beware renaming states_all/optim
    end
    optim.negLL = sum([states_all.negLL]);
    optim.AIC = 2*optim.negLL + 2*numel(model.params)*n_blocks;
    optim.BIC = 2*optim.negLL + log(sum(subj_data.n_trials)-sum(subj_data.missing,'all'))*numel(model.params)*n_blocks;
    optim.per_block = states_all;
else
    [params,states_best,optim] = fit_data(subj_data,model,param_combs,config); % beware renaming states_all/optim
end

end

function block_data = extract_block_data(subj_data,i_block)

block_data = subj_data;
vars = fieldnames(block_data);
for i_var = 1:numel(vars)
    s = size(subj_data.(vars{i_var}));
    if isa(subj_data.(vars{i_var}),'double') && s(end) == subj_data.n_blocks
        % if prod(s) > 1
            switch size(s,2)
                case 2, block_data.(vars{i_var}) = subj_data.(vars{i_var})(:,i_block);
                case 3, block_data.(vars{i_var}) = subj_data.(vars{i_var})(:,:,i_block);
            end
        % end
    end
end

end

function [params,states_best,states_all] = fit_data(data,model,param_combs,config)

[params,states_best,states_all] = mn_fitModel(data,model,param_combs,config);

end
