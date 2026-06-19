function [param_combs,n_combs] = mn_getGridMat(varargin)

param_grid      = cell(size(varargin));
[param_grid{:}] = ndgrid(varargin{:});
param_combs     = cell2mat(cellfun(@(x) x(:),param_grid,'UniformOutput',false));
n_combs = size(param_combs,1);
        
end