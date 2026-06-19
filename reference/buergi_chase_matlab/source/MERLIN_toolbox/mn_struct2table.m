function data = mn_struct2table(data_struct,n_trials)
%
% convert data struct to table
% variable with unique entries will be copied to get size nx1
%
% opposite conversion with mn_table2struct()
%

if nargin < 2
    get_n_from_data = 1;
else
    get_n_from_data = 0;
end

% duplicate values with singular entries
vars = fieldnames(data_struct);
for i_subj = 1:numel(data_struct)
    if get_n_from_data
        n_trials = data_struct(i_subj).n_trials(1);
    end
    for i_var = 1:numel(vars)
        if size(data_struct(i_subj).(vars{i_var}),1) == 1
            data_struct(i_subj).(vars{i_var}) = repmat(data_struct(i_subj).(vars{i_var}),n_trials,1);
        end
    end
end
% data_struct = rmfield(data_struct,'n_trials');

data = table();
for i_subj = 1:numel(data_struct)

    if isfield(data_struct(i_subj),'n_blocks')
        n_blocks = data_struct(i_subj).n_blocks(1);
    else
        n_blocks = 1;
    end
    
    if n_blocks == 1

        data = [data; struct2table(data_struct(i_subj))];

    else

        new_block_data = struct2table(data_struct(i_subj)); % for constant values
        for i_block = 1:n_blocks
            for i_var = 1:numel(vars)
                curr_val = data_struct(i_subj).(vars{i_var});
                if size(curr_val,2) == n_blocks
                    new_block_data.(vars{i_var}) = data_struct(i_subj).(vars{i_var})(:,i_block); % replace block-specific data
                end
            end
            data = [data; new_block_data];
        end

    end

end
    
end
