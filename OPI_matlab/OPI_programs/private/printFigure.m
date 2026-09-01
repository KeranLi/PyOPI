function printFigure(filePath)
% Save current figure in a portrait pdf format using portrait orientation, 
% and filename based on the functon and figure number that originated
% the call. 
% Input argument:
% filePath: defines a path for the pdf file (default: current path)
% Notes: 
% 1) The -dpdf option for print appears to have a default 
% resolution for graphics of 600 dpi. 
% 2) Avoid the -vector option, which can cause Matlab to crash for 
% a complex graphic image. 

% Mark Brandon, Yale University, August, 2022

%% Initialize variables
if nargin==0, filePath = []; else, filePath = [filePath, '/']; end
%... Get name of function that issued this command 
s = dbstack;
%... Initialize figure properties
hF = gcf;
set(hF, 'InvertHardcopy', 'off', 'color', 'w');
forceLightExportStyle(hF);
%... Set 
orient(hF, 'portrait')
%... Construct filename
figFilename = sprintf('%s_Fig%02d', [filePath, s(end).name], hF.Number);
%... Invoke print command
print(hF, '-dpdf', '-bestfit', figFilename);

end

function forceLightExportStyle(hF)
% Keep exported figures readable when MATLAB is running with a dark theme.
set(findall(hF, 'Type', 'axes'), ...
    'XColor', 'k', 'YColor', 'k', 'ZColor', 'k');
set(findall(hF, 'Type', 'text'), 'Color', 'k');

hColorbar = findall(hF, 'Type', 'colorbar');
if ~isempty(hColorbar)
    set(hColorbar, 'Color', 'k');
end

hLegend = findall(hF, 'Type', 'legend');
if ~isempty(hLegend)
    set(hLegend, 'TextColor', 'k', 'Color', 'w', 'EdgeColor', 'k');
end

% Annotation objects do not all inherit defaultTextColor consistently.
hObjects = findall(hF);
for i = 1:numel(hObjects)
    if isprop(hObjects(i), 'TextColor')
        set(hObjects(i), 'TextColor', 'k');
    end
end
end
