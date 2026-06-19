function fit = mn_fit_config()
%
% settings for the MERLIN fitting loop
%

% objective
fit.objective = 'ML'; % 'ML','MAP'

% grid search
fit.use_grid  = 1;
fit.save_grid = 0; % turn off to save memory

% parallelization
fit.use_parfor = 1;

% optimization
fit.use_optim  = 1;
fit.optim_algo = 'NelderMead';
fit.optim_n_it = 4;
fit.type_init  = 'grid'; %'grid','prior','uniform'

% parameter check
fit.check_param = 'warning'; % 'warning','error','none'
fit.max_val = 1e3;

% % missing data
% fit.nanHandling = 'removeRow'; % 'removeRow','skipUpdate','updateStates'

% save output
fit.save_output = 1; % 'struct','table'
fit.output_folder = fullfile(cd,'results','new_fits'); %'E:\2024_CHASE\results\fits';
fit.save.timestamp = 1; % default format 'yy-mm-dd_HH-MM-SS'

end