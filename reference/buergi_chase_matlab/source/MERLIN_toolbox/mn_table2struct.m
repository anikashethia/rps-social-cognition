function data = mn_table2struct(data_table,subj_var,varargin)
%
% convert data table to scalar struct, sorted by subjID
%
% optional input arguments
% - 'remove_redundancy': reduce variables with only one unique value to size 1x1
% - 'exceptions': to specify variables that should be kept as vectors
%   even if the vector just consists of repetitions (e.g. for missing trial
%   indicator varialbes, requires the next input to be a struct of one or more
%   variable names
% - 'block_var': to sort data into different blocks (to be fitted separatedly),
%   requires the next input to be the name of the varialbe in the dataset
%
% opposite conversion with mn_struct2table()
%

if nargin < 2
    subj_var = 'subjID';
end
% if nargin < 3
%     remove_redundancy = false;
% end
if any(strcmp(varargin,'remove_redundancy'))
    

% move whole subjects into struct
data_table.n_trials = NaN(height(data_table),1);
subj = unique(data_table.(subj_var));
for i_subj = 1:numel(subj)
    idx = (data_table.(subj_var) == subj(i_subj));
    data(i_subj,1) = table2struct(data_table(idx,:),'ToScalar',true);
    data(i_subj,1).(subj_var) = subj(i_subj);
    data(i_subj,1).n_trials = sum(idx);
end

% remove repetitions of values that are the same within a given subject 
if any(strcmp(varargin,'remove_redundancy'))
    vars = fieldnames(data);
    vars(strcmp(vars,subj_var)) = [];
    vars(ismember(vars,varargin{find(strcmp(varargin,'exceptions'))+1})) = [];
    for i_subj = 1:numel(subj)
        for i_var = 1:numel(vars)
            if numel(unique(data(i_subj).(vars{i_var}))) == 1
                data(i_subj).(vars{i_var})(2:end) = [];
            else
                try
                    if all(isnan(data(i_subj).(vars{i_var})))
                        data(i_subj).(vars{i_var}) = NaN;
                    end
                end
            end
        end
    end
end

% split up data into different blocks (along the second dimension)
if any(strcmp(varargin,'block_var')) %nargin > 3 % sort into blocks
    block_var = varargin{find(strcmp(varargin,'block_var'))+1};
    vars = fieldnames(data);
    vars(strcmp(vars,block_var)) = []; % move block to end
    vars = [vars;{block_var}];
    for i_subj = 1:numel(subj)
        blocks = unique(data(i_subj).(block_var));
        n_trials = histcounts(data(i_subj).(block_var)); % if a run was aborted early
        for i_var = 1:numel(vars)
            if numel(data(i_subj).(vars{i_var})) > 1
                clear new_data
                for i_block = 1:numel(blocks)
                    idx = (data(i_subj).(block_var) == blocks(i_block));
%                     assert(n_trials==sum(idx),'%d',data(i_subj).subID); % <- done to identify subj with missing half
                    new_data(:,i_block) = [data(i_subj).(vars{i_var})(idx); NaN(max(n_trials)-sum(idx),1)];
                end
                data(i_subj).(vars{i_var}) = new_data;
            end
        end
        data(i_subj).n_trials = n_trials;
        data(i_subj).n_blocks = numel(blocks);
    end
    
    % within blocks, remove redundancy again
    if any(strcmp(varargin,'remove_redundancy'))
        vars = fieldnames(data);
        vars(strcmp(vars,subj_var)) = [];
        vars(ismember(vars,varargin{find(strcmp(varargin,'exceptions'))+1})) = [];
        for i_subj = 1:numel(subj)
            for i_var = 1:numel(vars)
                n = NaN(1,size(data(i_subj).(vars{i_var}),2));
                for i_block = 1:size(n,2)
                    n(i_block) = numel(unique(data(i_subj).(vars{i_var})(:,i_block)));
                end
                if all(n == 1) && ~strcmp(vars{i_var},'missing')
                    data(i_subj).(vars{i_var})(2:end,:) = [];
                else
                    try
                        if all(isnan(data(i_subj).(vars{i_var})))
                            data(i_subj).(vars{i_var}) = NaN;
                        end
                    end
                end
            end
        end
    end
    
end
    
end
