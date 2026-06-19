function param_vec = mn_sampleUniform(settings,i_var,n)

grid = settings.params(i_var).grid;
param_vec = rand(n,1)*(min(max(grid),10)-min(grid)) + min(grid); % max 10 for now

% grid = settings.params(i_var).grid;
% param_vec = rand(n,1)*(max(grid)-min(grid)) + min(grid);   

end