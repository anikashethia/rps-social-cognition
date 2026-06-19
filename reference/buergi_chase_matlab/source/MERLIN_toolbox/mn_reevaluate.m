function MNs = mn_reevaluate(MNs,SAVE)
%
% takes an existing fit structure and uses the parameter estimates therin to
% update the optim values and internal states
%
% use to get outputs after changing either in a way that doesn't change the 
% likelihood of the parameters and therefore does not require new fitting
%

if nargin < 2
    SAVE = 0;
end

% loop over models
for i_model = 1:numel(MNs)
    
    mn_printProgress(i_model,numel(MNs),'','count');
    
    if isempty(MNs(i_model).model.params)
        warning('Model %s contains no parameters - skipping it.',model.name);
        continue
    end

    model = MNs(i_model).model;
    config = MNs(i_model).config;
    params = {model.params.name};
    
    assert(~config.save_grid,'Cannot save grid if simply reevaluating ML/MAP estimates. Fit without optimization instead.');
    
    % identify objective function
    switch config.objective
        case 'ML'
            loss_fxn = model.loglik_fxn;
        case 'MAP'
            loss_fxn = @mn_LL2AP;
    end
    
    % loop over subjects
    for i_subj = 1:numel(MNs(i_model).subj)
        
        clearvars states optim
        
        % extract data and parameter estimates
        data = MNs(i_model).subj(i_subj).data;
        for i_param = 1:numel(params)
            p_best(:,i_param) = [MNs(i_model).subj(i_subj).params.(params{i_param})];
        end
        
        if data.n_blocks == 1%~config.fit_per_block
            
            [optim.negLL,states] = loss_fxn(data,model,p_best','fit'); % <- transpose p_best or not?
            optim.AIC = 2*optim.negLL + 2*numel(model.params);
            optim.BIC = 2*optim.negLL + log(data.n_trials-sum(data.missing))*numel(model.params);
            
        else
            
            n_blocks = data.n_blocks;
            for i_block = 1:n_blocks
                
                block_data = extract_block_data(data,i_block);        
                [negLL,states(1,i_block)] = loss_fxn(block_data,model,p_best(i_block,:),'fit');
                
                % compute outputs per block
                per_block(1,i_block).negLL = negLL;
                per_block(1,i_block).AIC = 2*negLL + 2*numel(model.params);
                per_block(1,i_block).BIC = 2*negLL + log(block_data.n_trials-sum(block_data.missing))*numel(model.params);
                
            end
            
            % compute outputs per subject (across blocks)
            optim.negLL = sum([per_block.negLL]);
            optim.AIC = 2*optim.negLL + 2*numel(model.params)*n_blocks;
            optim.BIC = 2*optim.negLL + log(sum(data.n_trials)-sum(data.missing,'all'))*numel(model.params)*n_blocks;
            optim.per_block = per_block;
            
        end
            
        MNs(i_model).subj(i_subj).states = states;
        MNs(i_model).subj(i_subj).optim = optim;
        
    end
    
    if SAVE
        MN = MNs(i_model);
        filename = sprintf('%s_fits.mat',model.name);
        save(fullfile(config.output_folder,filename),'MN');
    end
    
end

end

function block_data = extract_block_data(subj_data,i_block)

block_data = subj_data;
vars = fieldnames(block_data);
for i_var = 1:numel(vars)
    s = size(subj_data.(vars{i_var}));
    if prod(s) > 1
        switch size(s,2)
            case 2, block_data.(vars{i_var}) = subj_data.(vars{i_var})(:,i_block);
            case 3, block_data.(vars{i_var}) = subj_data.(vars{i_var})(:,:,i_block);
        end
    end
end

end
        