function data = mn_createOutputTable(MN)

n_subj = size(MN.subj,1);

data = table();
for i_subj = 1:n_subj
    
    if MN.subj(i_subj).data.n_blocks == 1

        new_data = mn_struct2table(MN.subj(i_subj).data);
        new_states = mn_struct2table(MN.subj(i_subj).states,MN.subj(i_subj).data.n_trials);
        new_param = mn_struct2table(MN.subj(i_subj).params,MN.subj(i_subj).data.n_trials);
        data = [data; [new_data new_states new_param]];

    else

        new_data = mn_struct2table(MN.subj(i_subj).data);
        new_states = table();
        new_param = table();

        for i_block = 1:MN.subj(i_subj).data.n_blocks
            n_trials = MN.subj(i_subj).data.n_trials(i_block);
            new_states = [new_states; mn_struct2table(MN.subj(i_subj).states(i_block),n_trials)];
            new_param = [new_param; mn_struct2table(MN.subj(i_subj).params(i_block),n_trials)];
        end

        data = [data; [new_data new_states new_param]];

    end
    
end

end