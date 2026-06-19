function [MN,out] = mn_compare(MN,varargin)
%
% performs model comparison amongst all models in the struct array MN:
% - done based on negLL, AIC, or BIC
% - fixed effects or random effects (using the VBA toolbox)
%

if nargin < 1 || ~any(strcmp(varargin,'level_of_analysis'))
    level_of_analysis = 'per_subj'; % per_block
else
    level_of_analysis = varargin{find(strcmp(varargin,'level_of_analysis'))+1};
end
if nargin < 1 || ~any(strcmp(varargin,'criteria'))
    criteria = {'AIC'}; % {'negLL','AIC','BIC'};
else
    criteria = varargin{find(strcmp(varargin,'criteria'))+1};
end
if nargin < 1 || ~any(strcmp(varargin,'flag_plot'))
    flag_plot = 1;
else
    flag_plot = varargin{find(strcmp(varargin,'flag_plot'))+1};
end
if nargin < 1 || ~any(strcmp(varargin,'flag_individual_plots'))
    flag_individual_plots = 0;
else
    flag_individual_plots = varargin{find(strcmp(varargin,'flag_plot'))+1};
end
if nargin < 1 || ~any(strcmp(varargin,'flag_binary_comparisons'))
    flag_binary_comparisons = 0;
else
    flag_binary_comparisons = varargin{find(strcmp(varargin,'flag_plot'))+1};
end

VBA_verbose = 0;
VBA_show_fig = 0;

% flag_plot = 0;
% flag_binary_comparisons = 0;

%% get data

fprintf('\n+++ Model comparison +++\n\n\n');

colors = get_matlab_colors();
sz = 1e3;
scrsz = get(0,'screensize');

% get data
fprintf('\bGetting data... \n');
n_models = numel(MN);
model_names = arrayfun(@(x) x.model.name,MN,'UniformOutput',false);
n_crit = numel(criteria);
for i_model = 1:n_models
    for i_crit = 1:n_crit
        switch level_of_analysis
            case 'per_subj'
                fit(:,i_model,i_crit) = cell2mat(arrayfun(@(x) x.optim.(criteria{i_crit}),MN(i_model).subj,'UniformOutput',false));
            case 'per_block'
                fits_per_subj = arrayfun(@(x) {x.optim.per_block.(criteria{i_crit})},MN(i_model).subj,'UniformOutput',false);
                fit(:,i_model,i_crit) = cell2mat([fits_per_subj{:}]);
        end
    end
end

models = arrayfun(@(mn) mn.model.name,MN,'UniformOutput',0);

% get covariate
if any(strcmp(varargin,'Covariate'))
    idx_cov = find(strcmp(varargin,'Covariate')) + 1;
    cov_name = varargin{idx_cov};
    cov_param = arrayfun(@(subj) subj.data.(cov_name),MN(1).subj);
    [~,idx_cov] = sort(cov_param);
end
    
% get grouping variable
if any(strcmp(varargin,'group'))
    grouping_name = varargin{strcmp(varargin,'group')+1};
    grouping_param = arrayfun(@(subj) subj.data.(grouping_name),MN(1).subj);
    flag_individual_plots = 0;
    
    % if analyzing single blocks, copy the grouping param variable accordingly
    if strcmp(level_of_analysis,'per_block')
        n_blocks_per_subj = arrayfun(@(subj) subj.data.n_blocks,MN(1).subj);
        grouping_param_blocks = cellfun(@(x,y) repmat({x},y,1),grouping_param,num2cell(n_blocks_per_subj),'UniformOutput',0);
        grouping_param = cat(1,grouping_param_blocks{:});
    end
    
    if any(strcmp(varargin,'Covariate'))
        error('Combination of group and covariate not implemented yet.');
    end
else
    grouping_param = [];
end

%% run comparisons

all_fits = fit;
groups = unique(grouping_param);

for i_group = 1:max(1,numel(groups))
    
    if any(strcmp(varargin,'group'))
        curr_group = groups{i_group};
        idx_subj = strcmp(curr_group,grouping_param);
        mn_printProgress(i_group,numel(groups),'Running comparisons... ');
    else
        idx_subj = logical(ones(size(all_fits,1),1));
        fprintf('Running comparison... \n\n');
    end
    fit = all_fits(idx_subj,:,:);

    % fixed effects comparison (FFX)
    means = mean(fit,1);
    sems = mn_sem(fit);
    
    out(i_group).fix.(criteria{i_crit}).fitValues = fit(:,:,i_crit);
    out(i_group).fix.(criteria{i_crit}).means = means(1,:,i_crit);
    out(i_group).fix.(criteria{i_crit}).sems = sems(1,:,i_crit);

    if flag_individual_plots
        figure('Position',[(scrsz(3)-sz)/2 (scrsz(4)-sz)/2 sz sz]);
        for i_crit = 1:n_crit
            subplot(n_crit,1,i_crit); hold on;
            [~,idx_sort] = sort(means(:,:,i_crit),'descend');
            b = barh(means(1,idx_sort,i_crit));
            sem = [means(:,:,i_crit) - sems(:,:,i_crit); means(:,:,i_crit) + sems(:,:,i_crit)];
            line(sem(:,idx_sort),repmat(1:n_models,2,1),'Color','k');
            xlabel(criteria{i_crit});
            yticks(1:n_models);
            yticklabels(model_names(idx_sort));
            set(gca,'TickLabelInterpreter','none')
            curr_lims = [min(sem(:)) max(sem(:))];
            xlim([curr_lims(1)-diff(curr_lims)/2 curr_lims(2)+diff(curr_lims)/2]);
        end
        sgtitle("Model comparison (FFX)");
    end

    % head-to-head comparison (only if just two models)
    if n_models == 2 && flag_binary_comparisons

        figure('Position',[(scrsz(3)-sz)/2 (scrsz(4)-sz)/2 sz sz]);
        fits = fit;

        ii = 1;
        for i_crit = 1:n_crit

                subplot(n_crit,2,ii); hold on;
                curr_fits = squeeze(fits(:,:,i_crit));
                curr_diff = curr_fits(:,2) - curr_fits(:,1);
                diff_pos  = (curr_diff > 0);
                diff_neg  = (curr_diff < 0);
                diff_zero = (curr_diff == 0);

                title({"favoring (1)"},'Interpreter','none');

                scatter(curr_fits(diff_pos,1),curr_fits(diff_pos,2));
                scatter(curr_fits(diff_neg,1),curr_fits(diff_neg,2));
                if any(diff_zero)
                    scatter(curr_fits(diff_neg,1),curr_fits(diff_neg,2));
                end

                curr_lims(1,:) = xlim;
                curr_lims(2,:) = ylim;
                new_lims = [min(curr_lims(:,1)),max(curr_lims(:,2))];
                xlim(new_lims); ylim(new_lims);
                plot(new_lims,new_lims);
                axis square

                x = new_lims(end) + diff(new_lims)*0.05;
                y = (new_lims(end) + new_lims(1)) / 2;
                t = text(x,y,{"favoring (2)"},'Rotation',270,'HorizontalAlignment','center',...
                         'FontSize',11,'FontWeight','bold','Interpreter','none');

                ylabel(criteria{i_crit},'FontWeight','bold','FontSize',12);

                ii = ii + 1;
                subplot(n_crit,2,ii); hold on;

                yline(0,'k--');
                if any(strcmp(varargin,'Covariate'))
                    scatter(cov_param(diff_pos),curr_diff(diff_pos));  
                    scatter(cov_param(diff_neg),curr_diff(diff_neg));  
                    if any(diff_zero)
                        scatter(cov_param(diff_zero),curr_diff(diff_zero));   
                    end
                    title("(2)-(1) vs " + cov_name,'Interpreter','none');
                    xlabel(cov_name,'Interpreter','none');
                else
                    diff_sorted = sort(curr_diff);
                    scatter(find(diff_sorted > 0),diff_sorted(diff_sorted > 0));  
                    scatter(find(diff_sorted < 0),diff_sorted(diff_sorted < 0));  
                    if any(diff_sorted == 0)
                        scatter(find(diff_sorted == 0),diff_sorted(diff_sorted == 0));   
                    end
                    xlim([0,numel(diff_sorted)]);
                title("(2)-(1), sorted by value");
                end
                ylims = [-max(abs(ylim)),max(abs(ylim))];
                ylim(ylims);

                xlims = xlim;
                x = xlims(end) + diff(xlims)*0.05;
                y_top = ylims(1) + diff(ylims)/4*3;
                y_bottom = ylims(1) + diff(ylims)/4*1;
                t = text(x,y_top,{"favoring (1)"},'Rotation',270,'HorizontalAlignment','center',...
                         'FontSize',11,'FontWeight','bold','Interpreter','none');
                t = text(x,y_bottom,{"favoring (2)"},'Rotation',270,'HorizontalAlignment','center',...
                         'FontSize',11,'FontWeight','bold','Interpreter','none');
                ii = ii + 1;

        end

        sgtitle({"Head-to-head comparison","",""},'FontWeight','bold');
        sgtitle({"","(1) " + model_names{1} + " vs","(2) " + model_names{2}},'FontSize',10,'Interpreter','none');

    end

    % random effects comparison (RFX)
    options.modelNames = model_names;
    options.verbose = VBA_verbose;
    options.DisplayWin = VBA_show_fig;
    for i_crit = 1:n_crit
        LME_hat = -fit(:,:,i_crit)'; % as VBA is expected LMEs, need to reverse the sign for AIC/BIC <- also _neg_LL
        if any(strcmp(varargin,'Covariate'))
            LME_hat = LME_hat(:,idx_cov);
        end
        options.figName = "VBA RFX-BMS (" + criteria{i_crit} + ")";
        [posterior, out1] = VBA_groupBMC(LME_hat,options);
        rfx{i_crit}.counts = posterior.a;
        rfx{i_crit}.attributions = posterior.r;
        rfx{i_crit}.pxp = out1.pxp';
        out(i_group).rand.(criteria{i_crit}).counts = posterior.a;
        out(i_group).rand.(criteria{i_crit}).attributions = posterior.r;
        out(i_group).rand.(criteria{i_crit}).pxp = out1.pxp';
    end

    if flag_individual_plots       
        fprintf('\bCreating figures... \n\n');
        figure('Position',[(scrsz(3)-sz*1.5)/2 (scrsz(4)-sz)/2 sz*1.5 sz]);
        iPlot = 0;
        for i_crit = 1:n_crit

            % population level: model frequency & protected exceedance probability
            iPlot = iPlot + 1;
            subplot(n_crit,2,iPlot); hold on;
            modelFreq = rfx{i_crit}.counts / sum(rfx{i_crit}.counts);
            [~,idx_sort] = sort(mean(modelFreq,2));
            barh(rfx{i_crit}.pxp(idx_sort),'BarWidth',0.8,'FaceAlpha',0.1,...
                 'FaceColor','k','EdgeAlpha',0.3);
            barh(modelFreq(idx_sort,:),'BarWidth',0.6,'FaceAlpha',0.6,'EdgeColor',colors(1,:));
            yticks(1:size(modelFreq,1));
            xlim([0,1]);
            xticks(0:0.2:1);
            yticks(1:n_models);
            yticklabels(model_names(idx_sort)');
            set(gca,'TickLabelInterpreter','none')
            xline(0,'k');
            ylabel({'',criteria{i_crit},''},'FontWeight','bold','FontSize',12);
            if iPlot < 2, title({"Population-level","(model frequency & PXP)"}); end
            if iPlot == n_crit*2-1   
                xlabel("Model frequency (blue), PXP (grey)");
            end  

            % subject level: model attributions
            sortByWinningModel = 0;

            iPlot = iPlot + 1;
            subplot(n_crit,2,iPlot); hold on;
            [~,idx_sort] = sort(rfx{i_crit}.counts);
            if sortByWinningModel
                [~,idxSubj] = sort(rfx{i_crit}.attributions(idx_sort(end),:),'descend');
            else
    %             idxSubj = 1:numel(MN(1).subj);
                idxSubj = 1:sum(idx_subj);
            end
            imagesc(rfx{i_crit}.attributions(idx_sort,idxSubj));
            cmap = interp1([0, 1], [1 1 1; get_matlab_colors(1)], linspace(0, 1, 64));
            colormap(cmap)
            yticks(1:n_models);
            yticklabels(model_names(idx_sort)');
            set(gca,'TickLabelInterpreter','none')
            xlim([1,size(rfx{i_crit}.attributions,2)]);
            colorbar
            if iPlot == 2, title({"Subject-level","(model attributions)"}); end
            if iPlot == n_crit*2
                if any(strcmp(varargin,'Covariate'))
                    xlabel("Subjects (sorted by " + cov_name + ")",'Interpreter','none');
                else
                    xlabel("Subjects");
                end
            end
        end
        sgtitle("Model comparison (RFX)");
    end
    
end

%% combined plot

if (numel(groups) > 1 || any(strcmp(varargin,'group'))) && flag_plot
    
    mn_printProgress(numel(groups)+1,numel(groups),'');
    fprintf('\bCreating figures... \n\n');
    
    if ~any(strcmp(varargin,'use_current_fig'))
        figure; sgtitle('Random effects model comparison');
    end
    
    for i_crit = 1:numel(criteria)

        if ~any(strcmp(varargin,'use_current_fig'))
            subplot(numel(criteria),1,i_crit); hold on;
            title(criteria{i_crit});
        end
        
        pxp = [];
        for i_group = 1:max(1,numel(groups))
            pxp(:,i_group) = out(i_group).rand.(criteria{i_crit}).pxp;
        end

        [~,idx_sort] = sort(mean(pxp,2)); % ascending, i.e. worst to best (as barh is plotting along y axis)
        
        % account for bottom-to-top ordering within groupings
        pxp = fliplr(pxp);
        dataset_labels = flipud(groups); % to account 

%         figure
        b = barh(pxp(idx_sort,:));

        transparency = [0.6 0.6 0.6];
        for i_b = 1:numel(b)
            if contains(dataset_labels(i_b),'1'), c = 1;
            elseif contains(dataset_labels(i_b),{'2a','2b'}), c = 2;
            elseif contains(dataset_labels(i_b),{'2c','2d','2e','2f'}), c = 3;
            else, c = i_b;
            end
            b(i_b).FaceColor = colors(c,:);
            b(i_b).FaceAlpha = transparency(c)-0.1;
            b(i_b).EdgeColor = colors(c,:);
            b(i_b).EdgeAlpha = transparency(c);
            b(i_b).LineWidth = 0.5;
            transparency(c) = transparency(c) + 0.1;
        end

        yticks(1:size(pxp,1));
        xlim([0,1]);
        yticks(1:numel(models));
        yticklabels(models(idx_sort)');
        set(gca,'TickLabelInterpreter','none');
        box off
        yticks();

        idx_legend = numel(b):-1:1; % to match color order in legend to plot
        legend(b(idx_legend),groups,'FontSize',12);

        xticks(0:0.2:1);    
        xlabel('Protected Exceedance Probability (PXP)','FontSize',15);

        a = get(gca,'XTickLabel');  
        % set(gca,'XTickLabel',a,'fontsize',14,'FontWeight','normal'); % ,'FontWeight','bold'
        set(gca,'XTickLabelMode','auto')
        % set(gca,'linewidth',1.5);
        
    end
    
end

%%
fprintf('Done!\n\n');

end
