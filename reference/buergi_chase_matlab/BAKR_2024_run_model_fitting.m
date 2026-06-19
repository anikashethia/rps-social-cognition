%
% Model fitting for: "A neural signature of adaptive mentalization"
% by Buergi, Aydogan, Konovalov, & Ruff (Nature Neuroscience, 2026)
%
% Using the MERLIN toolbox (v0.01)
%

%% set folder

project_folder = cd;
addpath(fullfile(project_folder,'source'));
addpath(fullfile(project_folder,'source','MERLIN_toolbox'));

%% model fitting

% get data
load(fullfile(project_folder,'data','behavioral_data.mat'));
data = mn_table2struct(data,'subjID','remove_redundancy','exceptions',{'choice_own','choice_other','missing'});

% fit CHASE model
model = BAKR_2024_CHASE_config('CH','fitted',3,'RW-freq');
fits = mn_fit(data,model);

% fit alternative stastic models
models = {BAKR_2024_CHASE_config('LK','fixed',0,'RW-reward'),...  % reinforcement learning
          BAKR_2024_CHASE_config('LK','fixed',0,'EWA-single'),... % self-tuning EWA
          BAKR_2024_CHASE_config('LK','fixed',0,'EWA-full'),...   % standard EWA
          BAKR_2024_CHASE_config('LK','fixed',1,'RW-freq')};      % fictitious play
fits(2:5) = mn_fit(data,models);

% fit alternative adaptive model
model = BAKR_2024_ToMk_config;
fits(6) = mn_fit(data,model);

%% postprocessing

% rename models
fit_labels = arrayfun(@(fit) fit.model.name,fits,'UniformOutput',0);
fits(contains(fit_labels,'_CH_')).model.name             = 'CHASE';
fits(contains(fit_labels,'max-1_RW-freq')).model.name    = 'Ficticious play';
fits(contains(fit_labels,'max-0_RW-reward')).model.name  = 'Reward learner';
fits(contains(fit_labels,'max-0_EWA-single')).model.name = 'Self-tuning EWA';
fits(contains(fit_labels,'max-0_EWA-full')).model.name   = 'Full EWA';

% save fits
MN = fits(1);
mkdir(fullfile(project_folder,'results'));
save(fullfile(project_folder,'results','fits_CHASE_struct.mat'),'MN');
save(fullfile(project_folder,'results','model_comparison.mat'),'fits');

% create output table for CHASE model
data = mn_createOutputTable(fits(1));
data = movevars(data,'alpha','Before','beta');
data = movevars(data,'gamma','Before','lambda');
save(fullfile(project_folder,'results','fits_CHASE_table.mat'),'data');

%% replication

% get data
load(fullfile(project_folder,'data','replication','behavioral_data.mat'));
fits_replication = mn_fit(data',{fits.model}');

% save CHASE fit
MN = fits_replication(1);
mkdir(fullfile(project_folder,'results','replication'));
save(fullfile(folders.project,'results','replication','replication_model_fits.mat'),'MN');

% combine
for i_model = 1:numel(fits)
    for i_subj = 1:numel(fits_replication(i_model).subj)
        fits_replication(i_model).subj(i_subj).data.dataset = {'2f: Bot, RPS-3 (fMRI)'};
    end
    fits(i_model).subj = [fits(i_model).subj; fits_replication(i_model).subj];
end

save(fullfile(project_folder,'results','model_comparison.mat'),'fits');

%% --------------------------------------------------------------------------- %
%                           SUPPLEMENTARY ANALYSES                             %
% --------------------------------------------------------------------------- %%

idx_fmri = find(arrayfun(@(subj) contains(subj.data.dataset,'2e'),fits(1).subj));

%% determining the learning rule

% get data
load(fullfile(project_folder,'data','behavioral_data.mat'));
data = mn_table2struct(data,'subjID','remove_redundancy','exceptions',{'choice_own','choice_other','missing'});
fits_LR(1) = fits(1); % RW-freq already fitted above

% fit alternative versions
clear models
models{1} = BAKR_2024_CHASE_config('CH','fitted',3,'RW-reward');
models{2} = BAKR_2024_CHASE_config('CH','fitted',3,'RW-regret');
models{3} = BAKR_2024_CHASE_config('CH','fitted',3,'RW-hybrid');
fits_LR(2:4) = mn_fit(data,models); 

% rename 
for i_fit = 1:numel(fits_LR)
    fits_LR(i_fit).model.name = fits_LR(i_fit).model.learning_rule(4:end);
end

save(fullfile(project_folder,'results','supplementary','model_comparison_LR.mat'),'fits_LR');

%% model recovery: learning rule

% simulate
sims = BAKR_2024_simulate_data(fits_LR,idx_fmri);

% fit all simulated data with all models
models = {fits_LR.model};
recovery_fits = mn_fit(sims',models);

save(fullfile(project_folder,'results','supplementary','model_recovery_LR.mat'),'sims','sim_fits');

%% model recovery: full model

% simulate
sims = BAKR_2024_simulate_data(fits,idx_fmri);

% fit all simulated data with all models
models = {fits.model};
sim_fits = mn_fit(sims',models);

save(fullfile(project_folder,'results','model_recovery.mat'),'sims','sim_fits');

%% parameter recovery

% remove excluded subjects
subjID_old = arrayfun(@(subj) subj.data.subjID_original,fits(1).subj(idx_fmri));
idx_fmri(any(subjID_old == [7,94],2)) = [];

% simulate
[sims,params] = deal([]);
for k = 0:3
    [new_sims,new_params] = BAKR_2024_simulate_data(fits(1),idx_fmri,k);
    params = [params; new_params];
    sims = [sims new_sims];
end

% fit
model = fits(1).model;
fits_est = mn_fit(sims',model);
estimates = arrayfun(@(sim) struct2array(sim.params)',fits_est.subj);

prec.model = model;
prec.params.est = [estimates{:}]';
prec.params.gen = params;
save(fullfile(project_folder,'results','supplementary','parameter_recovery.mat'),'sims','prec');

%% effect of lowering the upper bound on gamma

load(fullfile(project_folder,'data','behavioral_data.mat'));
data = mn_table2struct(data,'subjID','remove_redundancy','exceptions',{'choice_own','choice_other','missing'});

model = BAKR_2024_CHASE_config('CH','fitted',3,'RW-freq');
gamma = arrayfun(@(subj) subj.params.gamma,fits(1).subj(idx_fmri));
idx_data = idx_fmri(gamma > 2.5);

UBs = [Inf,10,5,2.5,1,0.5];
for i_UB = 1:numel(UBs)

    UB = UBs(i_UB);

    models{i_UB} = model;
    models{i_UB}.params(3).support = [0,UB];
    if UB == Inf, models{i_UB}.params(3).space = 'log'; end
    models{i_UB}.params(3).grid = linspace(1e-4,min(2,UB-0.01),5);
    models{i_UB}.name = ['CHASE_gamma_max_' num2str(UB)];
    
end

fits_bounded = mn_fit(data(idx_data),models);
save(fullfile(project_folder,'results','supplementary','fits_bounded.mat'),'fits_bounded');

%% level estimation in human-human data

load(fullfile(project_folder,'data','behavioral_data.mat'));
data = mn_table2struct(data,'subjID','block_var','block','remove_redundancy','exceptions',{'choice_own','choice_other','missing'});
data(~contains([data.dataset],'1')) = [];

model = BAKR_2024_CHASE_config('LK','fitted',3,'RW-freq');
fits_LK_perblock = mn_fit(data,model);

save(fullfile(project_folder,'results','supplementary','fits_LK_perblock.mat'),'fits_LK_perblock');
