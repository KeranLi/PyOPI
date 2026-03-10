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
%... Set 
orient(hF, 'portrait')
%... Construct filename
figFilename = sprintf('%s_Fig%02d', [filePath, s(end).name], hF.Number);
%... Invoke print command
print(hF, '-dpdf', '-bestfit', figFilename);

end
