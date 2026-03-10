function [runTitle, nSamples, axisLabelsAll, exponentsAll, ...
    lbAll, ubAll, mSet, epsilon0, ...
    solutionsAll, chiR2Best, nuBest, betaBest] ...
    = getSolutions(solutionsPath, solutionsFile)
% getSolutions Reads and parses an opiFit solutions file.
%
% Mark Brandon, Yale University, 2016-2019

%% Read and process header
%... Open solutions file
fid = fopen([solutionsPath, '/', solutionsFile], 'r', 'native', 'UTF-8');
if fid==-1, error('Could not open user-provided solutions file'); end
    
%... Get title
while ~feof(fid)
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%'), break, end
end
runTitle = strtrim(str);

%... Get number of samples
while ~feof(fid)
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%'), break, end
end
nSamples = cell2mat(textscan(str, '%f'));
if isempty(nSamples)
    error('Input lacks entry for number of samples');
end

%... Get axis labels
while ~feof(fid)
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%'), break, end
end
axisLabelsAll = string(str);
axisLabelsAll = split(axisLabelsAll, "|");

%... Get exponents for power-of-10 factoring of parameter values
% For example, x = 300 with exponent = 2, would plot as 3.
while ~feof(fid)
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%'), break, end
end
exponentsAll = textscan(str, '%s', 'Delimiter', '\t');
exponentsAll = str2double(exponentsAll{1});

%... Get lower and upper constraints for parameter search
while ~feof(fid)
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%'), break, end
end
lbAll = str2num(str);  %#ok<ST2NM>
while ~feof(fid)
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%'), break, end
end
ubAll = str2num(str);  %#ok<ST2NM>
if ~isnumeric(lbAll) || ~isnumeric(ubAll)
    error('Parameter constraints must be numeric');
end
if length(lbAll) ~= length(ubAll)
    error('Constraints for parameter search much have the same length')
end
nParameters = length(lbAll);

%... Size of search set
while ~feof(fid)
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%'), break, end
end
mSet = str2double(str);

%... Specified epsilon value for stopping criterion, epsilon0
while ~feof(fid)
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%'), break, end
end
epsilon0 = str2double(str);

%% Read and process candidate solutions
solutionsAll = [];
nSolutionsAll = 0;
nLines = 0;
while ~feof(fid)
    str = strip(fgetl(fid));
    % Check for null characters
    % A problem with Matlab at Yale HPC is that rarely a record line will
    % include a long string of null characters. This happens randomly,
    % at a rate of about 1 out of 5000 records. The null characters
    % are displayed using the character '¿'. This issue is treated 
    % here by reporting these problem records, and then stripping the nulls
    % and treating them as a normal record.
    % Turned off reporting for records with strings, Sept, 2022
    % if ~isempty(str) && strcmp(str(1), char(0))        
    %    fprintf('Nulls at line %d: %s\n', nLines, ...
    %        strrep(str, char(0), '¿'));
    % end
    % Strip null characters from beginning and end of line, if present
    str = strip(str, char(0));
    nLines = nLines + 1;
    % Skip lines that are blank or prefixed by '%' (comment lines)
    if isempty(str) || strcmp(str(1),'%'), continue, end
    switch true
        case strcmpi(str(1:3), 'END')
            % End of candidate solutions
            break
        otherwise
            % Read candidate solution
            solution = cell2mat(textscan(str, '%f', 'Delimiter', '\t'));
            % Check for complete solution vector and then add to full set
            if length(solution)==nParameters+3 ...
                    || length(solution)==nParameters+4
                nSolutionsAll = nSolutionsAll + 1;
                solutionsAll(nSolutionsAll,:) = solution;  %#ok<AGROW>
            end
    end
end
fclose(fid);

%... Trim first column (iteration number) from solutionsAll
solutionsAll = solutionsAll(:,2:end);
    
%% Retrieve best-fit solution
[~, iMin] = min(solutionsAll(:,1));
chiR2Best = solutionsAll(iMin,1);
nuBest = solutionsAll(iMin,2);
betaBest = solutionsAll(iMin,3:end);

end
