function opiSync_ClumpedSensitivitySectionOrigin(groupName)
% opiSync_ClumpedSensitivitySectionOrigin rewrites section-origin lines in
% existing clumped sensitivity run files so they stay fixed at the sample site.

if nargin < 1
    groupName = "";
end

groupName = lower(string(groupName));
rootScenario = fullfile(fileparts(mfilename('fullpath')), '..', 'scenarios', ...
    'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
sectionOrigin = defaultSectionOrigin();
groupDirs = {
    'sensitivity_parameter_local_clumped'
    'sensitivity_divide_shift_clumped'
    'sensitivity_proxy_clumped'
    };

if groupName ~= ""
    groupDirs = groupDirs(contains(groupDirs, groupName));
    if isempty(groupDirs)
        error('Unknown sensitivity group: %s', groupName);
    end
end

for iGroup = 1:numel(groupDirs)
    groupDir = fullfile(rootScenario, groupDirs{iGroup});
    if ~isfolder(groupDir)
        continue
    end
    caseDirs = dir(groupDir);
    caseDirs = caseDirs([caseDirs.isdir] & ~startsWith({caseDirs.name}, '.'));
    for iCase = 1:numel(caseDirs)
        caseDir = fullfile(caseDirs(iCase).folder, caseDirs(iCase).name);
        runFiles = dir(fullfile(caseDir, '*.run'));
        for iFile = 1:numel(runFiles)
            rewriteSectionOrigin(fullfile(caseDir, runFiles(iFile).name), sectionOrigin);
        end
        rewriteReadme(caseDir, sectionOrigin);
    end
end

fprintf('Updated section origin to %.5f, %.5f for existing sensitivity cases.\n', ...
    sectionOrigin(1), sectionOrigin(2));
end

function rewriteSectionOrigin(runFile, sectionOrigin)
lines = readlines(runFile, 'WhitespaceRule', 'preserve');
idx = findActiveLineIndices(lines);
lines(idx(10)) = sprintf('%.5f, %.5f', sectionOrigin(1), sectionOrigin(2));
writelines(lines, runFile);
end

function rewriteReadme(caseDir, sectionOrigin)
readmeFile = fullfile(caseDir, 'README.txt');
if ~isfile(readmeFile)
    return
end
text = fileread(readmeFile);
newLine = sprintf('Fixed section origin: %.5f, %.5f', ...
    sectionOrigin(1), sectionOrigin(2));
if contains(text, 'Fixed section origin:')
    text = regexprep(text, 'Fixed section origin:\s*[-+0-9\.]+,\s*[-+0-9\.]+', newLine);
else
    text = sprintf('%s\n%s\n', strtrim(text), newLine);
end

fid = fopen(readmeFile, 'w', 'native', 'UTF-8');
if fid == -1
    error('Could not update README: %s', readmeFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s', text);
end

function idx = findActiveLineIndices(lines)
idx = [];
for i = 1:numel(lines)
    str = strip(string(lines(i)));
    if strlength(str) == 0
        continue
    end
    if startsWith(str, "%")
        continue
    end
    idx(end+1) = i; %#ok<AGROW>
end
if numel(idx) < 16
    error('Run file format did not match expected active-line count.');
end
end

function sectionOrigin = defaultSectionOrigin()
sectionOrigin = [87.2, 32.9];
end
