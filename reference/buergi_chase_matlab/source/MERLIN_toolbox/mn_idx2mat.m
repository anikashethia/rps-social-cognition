function matrix = mn_idx2mat(idx_vec,width,varargin)
% 
% formerly idx2position
%

if nargin < 2
    width = max(idx_vec);
end

matrix = zeros(numel(idx_vec),width);
idx = sub2ind(size(matrix),1:size(idx_vec,1),idx_vec');
if any(isnan(idx)) 
    if ismember(varargin,'remove_nan')
        idx(isnan(idx)) = [];
    else
        error('Input vector contains NaN values. To ignore them for the output, pass ''remove_nan'' as additional input argument.');
    end
end

matrix(idx) = 1;

end