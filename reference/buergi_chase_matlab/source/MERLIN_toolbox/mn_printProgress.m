function mn_printProgress(i_it,n_it,text,type,n_updates_max)
%
% print the progress of a given loop by updating a printed output (rather than
% printing below the previous output)
%

if nargin < 3
    text = '';
end
if nargin < 4
    type = 'percent'; % 'percent' or 'count'
end
if nargin < 5
    n_updates_max = 100;
end

% identify at which iterations to update, given the maximum number of updates
it_updates = unique(round(linspace(1,n_it+1,n_updates_max)));

if any(i_it == it_updates) 
    
    idx_curr = find(i_it == it_updates);
    
    % adapt to type
    switch type
        case 'percent'
            perc = round((i_it-1)/n_it*100); % -1 because currently working on this one
            perc_old = round((it_updates(max(1,idx_curr-1))-1)/n_it*100); % last one that was displayed -1 (to match above)
            new_text = sprintf('%s%i%%',text,perc);
            n_del = 3 + numel(text) + floor(log10(max(1,perc_old)));
            n_end = n_it + 1;
        case 'count'
            new_text = sprintf('%s%i/%i',text,i_it,n_it);
            n_del = 4 + numel(text) + floor(log10(max(0,i_it-1))) + floor(log10(n_it));
            n_end = n_it;
    end
    
    % print or overwrite
    if i_it == 1
        fprintf('%s\n',new_text);
    else
        fprintf([repmat('\b',1,n_del),'%s\n'],new_text);
    end
    
    if i_it == n_end
        fprintf('\n');
    end
    
end

end
