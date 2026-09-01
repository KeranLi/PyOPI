function summary = opiRepair_SmoothGridTopographyFiles( ...
    rootScenario, experimentName)
% Remove non-grid variables from existing smooth-grid topography MAT files.

% OPI gridRead requires exactly two coordinate vectors and one grid matrix.

if nargin < 1 || strlength(string(rootScenario)) == 0
    rootScenario = fullfile(fileparts(mfilename('fullpath')), '..', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
end
if nargin < 2 || strlength(string(experimentName)) == 0
    experimentName = 'topography_north_south_grid_smooth';
end

experimentRoot = fullfile(char(string(rootScenario)), ...
    char(string(experimentName)));
manifestFile = fullfile(experimentRoot, 'design', 'case_manifest.csv');
if ~isfile(manifestFile)
    error('Smooth-grid case manifest not found: %s', manifestFile);
end
manifest = readtable(manifestFile, 'TextType', 'string');

n = height(manifest);
variableCount = nan(n, 1);
status = strings(n, 1);
for i = 1:n
    topoFile = fullfile(experimentRoot, 'calc_only', ...
        manifest.case_id(i), 'Tibet_Eocene_30Ma_topo.mat');
    if ~isfile(topoFile)
        status(i) = "missing";
        continue
    end
    topo = load(topoFile, 'lon', 'lat', 'hGrid');
    required = {'lon', 'lat', 'hGrid'};
    if ~all(isfield(topo, required))
        status(i) = "invalid_required_variables";
        continue
    end
    save(topoFile, '-struct', 'topo', '-v7.3');
    info = whos('-file', topoFile);
    variableCount(i) = numel(info);
    names = string({info.name});
    if variableCount(i) == 3 && all(ismember(string(required), names))
        status(i) = "repaired";
    else
        status(i) = "invalid_after_repair";
    end
end

summary = table(manifest.case_id, variableCount, status, ...
    'VariableNames', {'case_id', 'variable_count', 'status'});
analysisRoot = fullfile(experimentRoot, 'analysis');
if ~isfolder(analysisRoot)
    mkdir(analysisRoot);
end
writetable(summary, fullfile(analysisRoot, ...
    'topography_mat_repair_status.csv'));
if any(status ~= "repaired")
    error('%d of %d smooth-grid topography files could not be repaired.', ...
        sum(status ~= "repaired"), n);
end
fprintf('Repaired %d smooth-grid topography files under:\n%s\n', ...
    n, experimentRoot);
end
