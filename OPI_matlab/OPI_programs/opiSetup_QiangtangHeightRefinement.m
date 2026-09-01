function opiSetup_QiangtangHeightRefinement( ...
    rootScenario, experimentName, qMinM, qMaxM, qStepM)
% Create a fine Qiangtang-height matrix using east-west latitude bands.
% The cases are fixed-parameter inputs; this function performs no OPI run.

if nargin < 1 || strlength(string(rootScenario)) == 0
    rootScenario = fullfile(fileparts(mfilename('fullpath')), '..', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
end
if nargin < 2 || strlength(string(experimentName)) == 0
    experimentName = 'topography_qiangtang_height_refinement';
end
if nargin < 3 || isempty(qMinM)
    qMinM = 3500;
end
if nargin < 4 || isempty(qMaxM)
    qMaxM = 4500;
end
if nargin < 5 || isempty(qStepM)
    qStepM = 250;
end
if ~isscalar(qMinM) || ~isscalar(qMaxM) || ~isscalar(qStepM) || ...
        any(~isfinite([qMinM, qMaxM, qStepM])) || qStepM <= 0 || ...
        qMinM > qMaxM
    error('Invalid Qiangtang height range or step.');
end
qValues = qMinM:qStepM:qMaxM;
if abs(qValues(end) - qMaxM) > 1e-9
    error('Qiangtang height step must land exactly on the upper bound.');
end
rootScenario = char(string(rootScenario));
experimentName = char(string(experimentName));
experimentRoot = fullfile(rootScenario, experimentName);
designRoot = fullfile(experimentRoot, 'design');
calcRoot = fullfile(experimentRoot, 'calc_only');
makeDir(designRoot);
makeDir(calcRoot);

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
base = load(fullfile(rootScenario, 'Tibet_Eocene_30Ma_topo.mat'), ...
    'lon', 'lat', 'hGrid');
required = string({sourceRun, sourceBest, sampleFile, clumpedFile, divideFile});
for i = 1:numel(required)
    if ~isfile(required(i))
        error('Required baseline file not found: %s', required(i));
    end
end

cases = makeCases(qValues);
manifestFile = fullfile(designRoot, 'case_manifest.csv');
qcFile = fullfile(designRoot, 'topography_quality_control.csv');
manifestId = fopen(manifestFile, 'w');
qcId = fopen(qcFile, 'w');
if manifestId == -1 || qcId == -1
    error('Could not open refinement design tables.');
end
cleanup = onCleanup(@() closeFiles(manifestId, qcId)); %#ok<NASGU>
fprintf(manifestId, ['case_id,gangdese_target_m,qiangtang_target_m,' ...
    'valley_mode,valley_target_m,parameter_mode,spatial_geometry,status\n']);
fprintf(qcId, ['case_id,gangdese_core_mean_m,qiangtang_core_mean_m,' ...
    'valley_core_mean_m,max_elevation_m,finite_grid,target_check,status\n']);

for i = 1:numel(cases)
    c = cases(i);
    caseDir = fullfile(calcRoot, c.id);
    if isfolder(caseDir)
        error('Case already exists; refusing to overwrite: %s', caseDir);
    end
    makeDir(caseDir);
    makeDir(fullfile(caseDir, 'proxy_clumped'));
    hGrid = buildTopography(base.hGrid, base.lat, c);
    lon = base.lon; %#ok<NASGU>
    lat = base.lat; %#ok<NASGU>
    save(fullfile(caseDir, 'Tibet_Eocene_30Ma_topo.mat'), ...
        'lon', 'lat', 'hGrid', '-v7.3');
    copyfile(sampleFile, fullfile(caseDir, 'Tibet_Eocene_30Ma_samples.xlsx'));
    copyfile(divideFile, fullfile(caseDir, ...
        'Tibet_Eocene_30Ma_topo_divide_main.mat'));
    copyfile(clumpedFile, fullfile(caseDir, 'proxy_clumped', ...
        'clumped_temperature.xlsx'));

    runName = ['Tibet_Eocene_30Ma_OxygenClumped_HeightRefine_', c.id, '.run'];
    bestName = ['Tibet_Eocene_30Ma_OxygenClumped_HeightRefine_', c.id, '_Best.run'];
    writeRun(sourceRun, fullfile(caseDir, runName), caseDir, c.id, false);
    writeRun(sourceBest, fullfile(caseDir, bestName), caseDir, c.id, true);
    writeMetadata(caseDir, c, runName, bestName);

    gMean = coreMean(hGrid, base.lat, 29.15, 0.35);
    qMean = coreMean(hGrid, base.lat, 33.20, 0.45);
    vMean = coreMean(hGrid, base.lat, 31.30, 0.45);
    pass = abs(gMean - c.gangdese) <= 75 && abs(qMean - c.qiangtang) <= 75;
    if c.valleyMode == "V1500"
        pass = pass && abs(vMean - c.valley) <= 75;
    end
    status = "pass";
    if ~pass
        status = "review";
    end
    fprintf(manifestId, '%s,%g,%g,%s,%g,fixed_clumped_best,east_west_latitude_only,generated\n', ...
        c.id, c.gangdese, c.qiangtang, c.valleyMode, c.valley);
    fprintf(qcId, '%s,%.6f,%.6f,%.6f,%.6f,%d,%d,%s\n', ...
        c.id, gMean, qMean, vMean, max(hGrid(:)), ...
        all(isfinite(hGrid(:))), pass, status);
end

writeReadme(experimentRoot, numel(cases), qValues, qStepM);
fprintf('Created %d Qiangtang height-refinement cases under:\n%s\n', ...
    numel(cases), calcRoot);
end

function cases = makeCases(qValues)
cases = struct('id', {}, 'gangdese', {}, 'qiangtang', {}, ...
    'valleyMode', {}, 'valley', {});
k = 0;
for g = [3000, 4500]
    for q = qValues
        k = k + 1;
        cases(k) = struct('id', sprintf('G%d_Q%d_Vnone', g, q), ...
            'gangdese', g, 'qiangtang', q, ...
            'valleyMode', "Vnone", 'valley', nan);
    end
end
for g = [3000, 4500]
    k = k + 1;
    cases(k) = struct('id', sprintf('G%d_Q4000_V1500', g), ...
        'gangdese', g, 'qiangtang', 4000, ...
        'valleyMode', "V1500", 'valley', 1500);
end
end

function hGrid = buildTopography(hGrid, lat, c)
hGrid = imposeBand(hGrid, lat, 29.15, 0.35, 0.45, c.gangdese);
hGrid = imposeBand(hGrid, lat, 33.20, 0.45, 0.45, c.qiangtang);
if c.valleyMode == "V1500"
    hGrid = imposeBand(hGrid, lat, 31.30, 0.45, 0.55, c.valley);
else
    hGrid = imposeRamp(hGrid, lat, c.gangdese, c.qiangtang);
end
hGrid = max(hGrid, 0);
end

function hGrid = imposeBand(hGrid, lat, center, halfWidth, transition, target)
w = bandWeight(lat, center, halfWidth, transition);
for i = 1:size(hGrid, 1)
    if w(i) > 0
        hGrid(i, :) = (1 - w(i)) .* hGrid(i, :) + w(i) .* target;
    end
end
end

function hGrid = imposeRamp(hGrid, lat, southTarget, northTarget)
center = 31.30;
halfWidth = 0.45;
transition = 0.55;
w = bandWeight(lat, center, halfWidth, transition);
south = center - halfWidth - transition;
north = center + halfWidth + transition;
for i = 1:size(hGrid, 1)
    if w(i) > 0
        f = min(max((lat(i) - south) / (north - south), 0), 1);
        target = southTarget + f * (northTarget - southTarget);
        hGrid(i, :) = (1 - w(i)) .* hGrid(i, :) + w(i) .* target;
    end
end
end

function w = bandWeight(lat, center, halfWidth, transition)
d = abs(lat(:) - center);
w = zeros(size(d));
w(d <= halfWidth) = 1;
mask = d > halfWidth & d < halfWidth + transition;
x = (d(mask) - halfWidth) ./ transition;
w(mask) = 1 - x .* x .* (3 - 2 .* x);
end

function value = coreMean(hGrid, lat, center, halfWidth)
mask = abs(lat(:) - center) <= halfWidth;
value = mean(hGrid(mask, :), 'all', 'omitnan');
end

function writeRun(sourceFile, targetFile, caseDir, caseId, isBest)
lines = readlines(sourceFile, 'WhitespaceRule', 'preserve');
idx = activeLines(lines);
if isBest
    suffix = ' fixed best-fit calculation';
else
    suffix = ' joint oxygen plus clumped refit template';
end
lines(idx(1)) = "Qiangtang height refinement " + caseId + suffix;
lines(idx(3)) = string(caseDir);
writelines(lines, targetFile);
end

function writeMetadata(caseDir, c, runName, bestName)
fid = fopen(fullfile(caseDir, 'case_metadata.csv'), 'w');
if fid == -1
    error('Could not write metadata: %s', caseDir);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'field,value\n');
fprintf(fid, 'case_id,%s\n', c.id);
fprintf(fid, 'gangdese_target_m,%g\n', c.gangdese);
fprintf(fid, 'qiangtang_target_m,%g\n', c.qiangtang);
fprintf(fid, 'valley_mode,%s\n', c.valleyMode);
fprintf(fid, 'valley_target_m,%g\n', c.valley);
fprintf(fid, 'parameter_mode,fixed_clumped_best\n');
fprintf(fid, 'spatial_geometry,east_west_latitude_only\n');
fprintf(fid, 'primary_refit_template,%s\n', runName);
fprintf(fid, 'fixed_calculation_run,%s\n', bestName);
fprintf(fid, 'target_precipitation_d18O_permil,-13.54\n');
fprintf(fid, 'primary_spatial_support,50km_precipitation_weighted\n');
fprintf(fid, 'divide_interpretation,not_used_as_causal_factor\n');
end

function writeReadme(experimentRoot, count, qValues, qStepM)
fid = fopen(fullfile(experimentRoot, 'README.md'), 'w');
if fid == -1
    error('Could not write refinement README.');
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Qiangtang Height Refinement\n\n');
fprintf(fid, ['Cases generated: **%d**. Gangdese candidates are 3000 and ' ...
    '4500 m; Qiangtang is sampled every %.0f m from %.0f to %.0f m.\n\n'], ...
    count, qStepM, qValues(1), qValues(end));
fprintf(fid, 'The primary comparison is 50 km precipitation-weighted d18O against the independent target of -13.54 per mil.\n');
end

function idx = activeLines(lines)
idx = [];
for i = 1:numel(lines)
    s = strip(string(lines(i)));
    if strlength(s) > 0 && ~startsWith(s, "%")
        idx(end + 1) = i; %#ok<AGROW>
    end
end
if numel(idx) < 10
    error('Run file does not contain expected active lines.');
end
end

function makeDir(pathName)
if ~isfolder(pathName)
    mkdir(pathName);
end
end

function closeFiles(varargin)
for i = 1:nargin
    if varargin{i} ~= -1
        fclose(varargin{i});
    end
end
end
