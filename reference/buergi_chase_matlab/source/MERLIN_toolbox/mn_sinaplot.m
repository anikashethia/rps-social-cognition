function ax = mn_sinaplot(data,bins,x_center,color,dot_size,scaling_factor,error_type,density_type,y_jitter)

% defaults
if nargin < 3 || isempty(x_center)
    x_center = 1;
end
if nargin < 4 || isempty(color)
    color = [0 0.4470 0.7410];
end
if nargin < 5 || isempty(dot_size)
    dot_size = 46;
end
if nargin < 6 || isempty(scaling_factor)
    scaling_factor = 1;
end
if nargin < 7 || isempty(error_type)
    error_type = 'sem';
end
if nargin < 8 || isempty(density_type)
    density_type = 'mirrored';
end
if nargin < 9 || isempty(y_jitter)
    y_jitter = 0;
end

% get density and bins
if nargin < 2 || isempty(bins)
    [density,bins] = ksdensity(data(:));
else
    density = ksdensity(data(:),bins);
end

% scale
overall_density = density * scaling_factor;
density = overall_density * 1.4;

% loop through data points and sample x pos (from range given by density)
x_pos = NaN(size(data));
for i_y = 1:numel(data)
    idx_bin = find(data(i_y) > bins,1,'last');
    if ~isempty(idx_bin)
        switch density_type
            case 'left',     offset = -rand * density(idx_bin);
            case 'right',    offset = rand * density(idx_bin);
            case 'mirrored', offset = rand * density(idx_bin) - density(idx_bin)/2;
        end
        x_pos(i_y) = x_center + offset;
    end
end

% prepare density
hold on;
density = overall_density;
poly_x = density;
poly_y = bins;

% remove empty parts
idx_del = (density < 1e-4);
poly_x(idx_del) = [];
poly_y(idx_del) = [];

switch density_type
    case 'mirrored'
        poly_x = [x_center + poly_x, x_center - fliplr(poly_x)];
        poly_y = [poly_y fliplr(poly_y)];
    otherwise
        poly_x = [0 density 0] + x_center;
end

ax.density = fill(poly_x,...
         poly_y,...
         color,...
         'EdgeColor',color,...
         'EdgeAlpha',0.8,...
         'FaceAlpha',0.2);   

% add individual datapoints
if y_jitter > 0
    y_pos = data + (rand(size(data))*2-1) * 0.25;
else
    y_pos = data;
end
ax.datapoints = scatter(x_pos,y_pos,...
            'SizeData',dot_size,'MarkerFaceColor','flat','CData',color,...
            'MarkerFaceAlpha',0.4,...
            'MarkerEdgeAlpha',0.4);

% add mean and variability (std or sem)
switch error_type
    case 'std', err = nanstd(data);
    case 'sem', err = nanstd(data)./sqrt(sum(~isnan(data)));
end

e = errorbar(x_center,nanmean(data),err,':','Color',[0.1 0.1 0.1 0.8],'LineWidth',dot_size*0.1); % 0.075
e.CapSize = dot_size*0.1;
% s2 = plot([x_center-dot_size*0.002,x_center+dot_size*0.002],[nanmean(data),nanmean(data)],'k','LineWidth',2.5);%dot_size*0.05);
s2 = scatter(x_center,nanmean(data),'SizeData',dot_size*1); % dot_size*1.75 - 75
% s2.CData = color;
s2.MarkerFaceColor = color;
s2.MarkerEdgeColor = 'k';
s2.LineWidth = dot_size*0.05; % 0.075
ax.mean = s2;
ax.errorbar = e;

end
