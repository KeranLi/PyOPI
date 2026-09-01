function opiSetup_NorthSouthElevationGridSmooth( ...
    rootScenario, experimentName, qValuesM, valleyValuesM, gangdeseValuesM)
% Build a smooth Qiangtang-valley-Gangdese terrain ensemble.
%
% Regional target elevations define a continuous PCHIP zonal-mean profile.
% The pre-platform source terrain's two-dimensional residual relief is then
% added back, avoiding flat latitude bands and low-elevation seams.

if nargin < 1 || strlength(string(rootScenario)) == 0
    rootScenario = fullfile(fileparts(mfilename('fullpath')), '..', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
end
if nargin < 2 || strlength(string(experimentName)) == 0
    experimentName = 'topography_north_south_grid_smooth';
end
if nargin < 3 || isempty(qValuesM)
    qValuesM = 3500:500:5500;
end
if nargin < 4 || isempty(valleyValuesM)
    valleyValuesM = 1500:500:5000;
end
if nargin < 5 || isempty(gangdeseValuesM)
    gangdeseValuesM = 3000:500:4500;
end
validateValues(qValuesM, 'Qiangtang');
validateValues(valleyValuesM, 'central valley');
validateValues(gangdeseValuesM, 'Gangdese');

rootScenario = char(string(rootScenario));
experimentRoot = fullfile(rootScenario, char(string(experimentName)));
if isfolder(experimentRoot)
    error('Smooth-grid experiment already exists; refusing to overwrite: %s', ...
        experimentRoot);
end
designRoot = fullfile(experimentRoot, 'design');
calcRoot = fullfile(experimentRoot, 'calc_only');
mkdir(designRoot);
mkdir(calcRoot);

sourceScenario = fullfile(fileparts(rootScenario), ...
    'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid');
sourceTopoFile = fullfile(sourceScenario, 'Tibet_Eocene_30Ma_topo.mat');
sourceDir = fullfile(rootScenario, 'oxygen_clumped_ultra_aggressive');
sourceRun = fullfile(sourceDir, ...
    'Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive.run');
sourceBest = fullfile(sourceDir, ...
    'Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive_Best.run');
sampleFile = fullfile(rootScenario, 'Tibet_Eocene_30Ma_samples.xlsx');
clumpedFile = fullfile(rootScenario, 'proxy_clumped', ...
    'clumped_temperature.xlsx');
divideFile = fullfile(rootScenario, ...
    'Tibet_Eocene_30Ma_topo_divide_main.mat');
required = string({sourceTopoFile, sourceRun, sourceBest, sampleFile, ...
    clumpedFile, divideFile});
missing = required(~isfile(required));
if ~isempty(missing)
    error('Missing smooth-grid input(s):\n%s', strjoin(missing, newline));
end
base = load(sourceTopoFile, 'lon', 'lat', 'hGrid');

cases = makeCases(qValuesM, valleyValuesM, gangdeseValuesM);
manifestFile = fullfile(designRoot, 'case_manifest.csv');
qcFile = fullfile(designRoot, 'topography_quality_control.csv');
profileFile = fullfile(designRoot, 'zonal_mean_profiles.csv');
manifestId = fopen(manifestFile, 'w');
qcId = fopen(qcFile, 'w');
profileId = fopen(profileFile, 'w');
if manifestId == -1 || qcId == -1 || profileId == -1
    error('Could not open smooth-grid design tables.');
end
cleanup = onCleanup(@() closeFiles(manifestId, qcId, profileId)); %#ok<NASGU>
fprintf(manifestId, ['case_id,gangdese_target_m,qiangtang_target_m,' ...
    'valley_mode,valley_target_m,parameter_mode,spatial_geometry,status\n']);
fprintf(qcId, ['case_id,gangdese_core_mean_m,qiangtang_core_mean_m,' ...
    'valley_core_mean_m,min_elevation_m,max_elevation_m,' ...
    'max_core_target_error_m,finite_grid,no_interregional_seam,status\n']);
fprintf(profileId, 'case_id,latitude_degN,zonal_mean_elevation_m\n');

for i = 1:numel(cases)
    c = cases(i);
    caseDir = fullfile(calcRoot, c.id);
    mkdir(caseDir);
    mkdir(fullfile(caseDir, 'proxy_clumped'));
    [hGrid, targetProfile] = buildSmoothTopography(base.hGrid, base.lat, c);
    lon = base.lon; %#ok<NASGU>
    lat = base.lat; %#ok<NASGU>
    save(fullfile(caseDir, 'Tibet_Eocene_30Ma_topo.mat'), ...
        'lon', 'lat', 'hGrid', '-v7.3');
    copyfile(sampleFile, fullfile(caseDir, 'Tibet_Eocene_30Ma_samples.xlsx'));
    copyfile(divideFile, fullfile(caseDir, ...
        'Tibet_Eocene_30Ma_topo_divide_main.mat'));
    copyfile(clumpedFile, fullfile(caseDir, 'proxy_clumped', ...
        'clumped_temperature.xlsx'));

    runName = ['Tibet_Eocene_30Ma_SmoothGrid_', c.id, '.run'];
    bestName = ['Tibet_Eocene_30Ma_SmoothGrid_', c.id, '_Best.run'];
    writeRun(sourceRun, fullfile(caseDir, runName), caseDir, c.id, false);
    writeRun(sourceBest, fullfile(caseDir, bestName), caseDir, c.id, true);
    writeMetadata(caseDir, c, runName, bestName, sourceTopoFile);

    gMean = coreMean(hGrid, base.lat, 29.15, 0.35);
    vMean = coreMean(hGrid, base.lat, 31.30, 0.45);
    qMean = coreMean(hGrid, base.lat, 33.20, 0.45);
    targetError = max(abs([gMean - c.gangdese, ...
        vMean - c.valley, qMean - c.qiangtang]));
    noSeam = assessNoArtificialSeam(base.lat, targetProfile, c);
    pass = targetError <= 1e-8 && all(isfinite(hGrid), 'all') && noSeam;
    status = "pass";
    if ~pass
        status = "review";
    end
    fprintf(manifestId, ...
        '%s,%g,%g,%s,%g,fixed_clumped_best,continuous_pchip_zonal_mean_with_2d_residual,generated\n', ...
        c.id, c.gangdese, c.qiangtang, c.valleyMode, c.valley);
    fprintf(qcId, '%s,%.9f,%.9f,%.9f,%.9f,%.9f,%.12g,%d,%d,%s\n', ...
        c.id, gMean, qMean, vMean, min(hGrid, [], 'all'), ...
        max(hGrid, [], 'all'), targetError, ...
        all(isfinite(hGrid), 'all'), noSeam, status);
    profile = mean(hGrid, 2, 'omitnan');
    for j = 1:numel(base.lat)
        fprintf(profileId, '%s,%.9f,%.9f\n', c.id, base.lat(j), profile(j));
    end
end

writeReadme(experimentRoot, sourceTopoFile, qValuesM, valleyValuesM, ...
    gangdeseValuesM, numel(cases));
fprintf('Created %d smooth north-south cases under:\n%s\n', ...
    numel(cases), calcRoot);
end

function validateValues(values, label)
if ~isvector(values) || isempty(values) || any(~isfinite(values)) || ...
        any(values < 0) || numel(unique(values)) ~= numel(values)
    error('%s elevation values must be unique finite nonnegative values.', label);
end
end

function cases = makeCases(qValues, valleyValues, gangdeseValues)
cases = struct('id', {}, 'gangdese', {}, 'qiangtang', {}, ...
    'valleyMode', {}, 'valley', {});
k = 0;
for g = sort(gangdeseValues)
    for v = sort(valleyValues)
        for q = sort(qValues)
            k = k + 1;
            cases(k) = struct( ...
                'id', sprintf('G%d_Q%d_V%d', g, q, v), ...
                'gangdese', g, 'qiangtang', q, ...
                'valleyMode', "V" + string(v), 'valley', v);
        end
    end
end
end

function [hGrid, targetProfile] = buildSmoothTopography(hGrid, lat, c)
zonalMean = mean(hGrid, 2, 'omitnan');
residualRelief = hGrid - zonalMean;
nodeLat = [lat(1); 29.15 - 0.35; 29.15 + 0.35; ...
    31.30 - 0.45; 31.30 + 0.45; ...
    33.20 - 0.45; 33.20 + 0.45; lat(end)];
nodeElevation = [zonalMean(1); c.gangdese; c.gangdese; ...
    c.valley; c.valley; c.qiangtang; c.qiangtang; zonalMean(end)];
if any(diff(nodeLat) <= 0)
    error('Smooth terrain control latitudes must be strictly increasing.');
end
targetProfile = interp1(nodeLat, nodeElevation, lat(:), 'pchip');
hGrid = residualRelief + targetProfile;
hGrid = max(hGrid, 0);
end

function value = coreMean(hGrid, lat, center, halfWidth)
mask = abs(lat(:) - center) <= halfWidth;
value = mean(hGrid(mask, :), 'all', 'omitnan');
end

function pass = assessNoArtificialSeam(lat, targetProfile, c)
% A monotone PCHIP transition must remain between its adjacent targets.
segments = [29.15 + 0.35, 31.30 - 0.45, c.gangdese, c.valley; ...
    31.30 + 0.45, 33.20 - 0.45, c.valley, c.qiangtang];
pass = true;
for i = 1:size(segments, 1)
    mask = lat >= segments(i, 1) & lat <= segments(i, 2);
    lower = min(segments(i, 3:4));
    upper = max(segments(i, 3:4));
    values = targetProfile(mask);
    pass = pass && all(values >= lower - 1e-8 & values <= upper + 1e-8);
end
end

function writeRun(sourceFile, targetFile, caseDir, caseId, isBest)
lines = readlines(sourceFile, 'WhitespaceRule', 'preserve');
idx = activeLines(lines);
suffix = ' smooth-terrain refit template';
if isBest
    suffix = ' smooth-terrain fixed calculation';
end
lines(idx(1)) = "Smooth north-south elevation grid " + caseId + suffix;
lines(idx(3)) = string(caseDir);
writelines(lines, targetFile);
end

function idx = activeLines(lines)
idx = [];
for i = 1:numel(lines)
    value = strip(string(lines(i)));
    if strlength(value) > 0 && ~startsWith(value, "%")
        idx(end + 1) = i; %#ok<AGROW>
    end
end
if numel(idx) < 10
    error('Run file does not contain expected active lines.');
end
end

function writeMetadata(caseDir, c, runName, bestName, sourceTopoFile)
T = table(["case_id"; "gangdese_target_m"; "qiangtang_target_m"; ...
    "valley_mode"; "valley_target_m"; "parameter_mode"; ...
    "spatial_geometry"; "source_topography"; "primary_refit_template"; ...
    "fixed_calculation_run"], ...
    [string(c.id); string(c.gangdese); string(c.qiangtang); ...
    string(c.valleyMode); string(c.valley); "fixed_clumped_best"; ...
    "continuous_pchip_zonal_mean_with_2d_residual"; ...
    string(sourceTopoFile); string(runName); string(bestName)], ...
    'VariableNames', {'field', 'value'});
writetable(T, fullfile(caseDir, 'case_metadata.csv'));
end

function writeReadme(root, sourceTopo, q, v, g, count)
fid = fopen(fullfile(root, 'README.md'), 'w');
if fid == -1
    error('Could not write smooth-grid README.');
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Smooth North-South Elevation Grid\n\n');
fprintf(fid, 'Cases: **%d** combinations.\n\n', count);
fprintf(fid, 'Source terrain: `%s`\n\n', sourceTopo);
fprintf(fid, '- Qiangtang m: `%s`\n', strjoin(string(q), ', '));
fprintf(fid, '- Central valley m: `%s`\n', strjoin(string(v), ', '));
fprintf(fid, '- Gangdese m: `%s`\n\n', strjoin(string(g), ', '));
fprintf(fid, ['Regional cores retain their requested zonal-mean elevations. ' ...
    'PCHIP transitions connect adjacent cores continuously, and the ' ...
    'pre-platform terrain residual preserves two-dimensional relief. ' ...
    'No transition returns to the low source terrain between regions.\n']);
end

function closeFiles(varargin)
for i = 1:nargin
    if varargin{i} ~= -1
        fclose(varargin{i});
    end
end
end
