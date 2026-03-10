function pathOld = fnPersistentPath(pathNew)
%... Set and retrieve a persistent path, using the base workspace.
% The path name is returned without a terminal slash. This convention 
% is preferred to maintain consistency with different operating systems.
%
% Example:
% dataPath = fnPersistentPath;
% [dataFile, dataPath] = uigetfile([dataPath, '/*.*']);
% if dataFile==0, error('No data file selected'), end
% fnPersistentPath(dataPath);

% Mark Brandon, Yale University, January 2019

%% Calculate
switch nargin
    case 0
        % Retrieve existing persistent path, if present
        try
            pathOld = evalin('base', 'persistentPath');
        catch ME
            if strcmp(ME.identifier, 'MATLAB:UndefinedFunction')
                % pwd returns current path name without a terminal slash.
                pathOld = pwd;
            end
        end
    case 1
        % Set new persistent path
        % Remove terminal slash, if present
        if pathNew(end)=='/' || pathNew(end)=='\'
            pathNew = pathNew(1:end-1); 
        end
        assignin('base', 'persistentPath', pathNew)
end

end
