function [sims,params] = BAKR_2024_simulate_data(fits,idx,k)

task = mn_RPS_config;

for i_model = 1:numel(fits)

    curr_model = fits(i_model).model;
    curr_model.sim_fxn = curr_model.loglik_fxn;

    for i_subj = 1:numel(idx)

        % extract subj-specific infos
        task.bot.levels = fits(1).subj(idx(i_subj)).data.bot_level(1:40:end);
        curr_params = struct2array(fits(i_model).subj(idx(i_subj)).params)';
        
        % allow for overriding of level parameter
        if nargin > 1
            curr_params(end) = k;
            params(i_subj,:) = curr_params;
            subjID = k*100 + i_subj;
        else
            subjID = i_model*100 + i_subj;
        end

        % simulate
        sim = mn_sim(task,curr_model,curr_params);
        sim.subj.data.subjID = subjID;
        sim.subj.data.model = fits(i_model).model.name;
        sim.subj.data.loss = -1;

        % add
        if i_model == 1 && i_subj == 1
            sims(1) = sim.subj.data;
        else
            sims(end+1) = sim.subj.data;
        end

    end

end

end