function opiSetup_TopographySensitivityCases(rootScenario, experimentName)
% Create topography case skeletons using the oxygen plus clumped workflow.
% This function only copies inputs and writes metadata; it never runs a fit.

if nargin < 1 || strlength(string(rootScenario)) == 0
    rootScenario = fullfile(fileparts(mfilename('fullpath')), '..', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
end
rootScenario = char(string(rootScenario));
if nargin < 2 || strlength(string(experimentName)) == 0
    experimentName = 'topography_sensitivity_clumped';
end
experimentName = char(string(experimentName));
experimentRoot = fullfile(rootScenario, experimentName);
if ~isfolder(experimentRoot)
    mkdir(experimentRoot);
end
designRoot = fullfile(experimentRoot, 'design');
if ~isfolder(designRoot)
    mkdir(designRoot);
end
calcRoot = fullfile(experimentRoot, 'calc_only');
if ~isfolder(calcRoot)
    mkdir(calcRoot);
end

sourceScenario = fullfile(fileparts(rootScenario), ...
    'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid');
clumpedRun = fullfile(rootScenario, 'oxygen_clumped_ultra_aggressive', ...
    'Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive.run');
clumpedBestRun = fullfile(rootScenario, 'oxygen_clumped_ultra_aggressive', ...
    'Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive_Best.run');
sampleFile = fullfile(rootScenario, 'Tibet_Eocene_30Ma_samples.xlsx');
clumpedFile = fullfile(rootScenario, 'proxy_clumped', ...
    'clumped_temperature.xlsx');
divideFile = fullfile(rootScenario, 'Tibet_Eocene_30Ma_topo_divide_main.mat');

requiredFiles = [string(clumpedRun); string(clumpedBestRun); ...
    string(sampleFile); string(clumpedFile); string(divideFile)];
for i = 1:numel(requiredFiles)
    if ~isfile(requiredFiles(i))
        error('Required baseline file not found: %s', requiredFiles(i));
    end
end

base = load(fullfile(rootScenario, 'Tibet_Eocene_30Ma_topo.mat'), ...
    'lon', 'lat', 'hGrid');
original = load(fullfile(sourceScenario, 'Tibet_Eocene_30Ma_topo.mat'), ...
    'lon', 'lat', 'hGrid');
if ~isequal(size(base.hGrid), size(original.hGrid))
    error('Baseline and source topography grids have different sizes.');
end

patterns = buildPatterns(base, original);
heightLevels = [3000, 4150, 5000];
normalizationModes = ["Mfixed", "Nmfixed"];

manifestId = fopen(fullfile(designRoot, 'case_manifest.csv'), 'w');
qcId = fopen(fullfile(designRoot, ...
    'topography_quality_control.csv'), 'w');
if manifestId == -1 || qcId == -1
    error('Could not open experiment design output files.');
end
cleanup = onCleanup(@() closeFiles(manifestId, qcId)); %#ok<NASGU>
fprintf(manifestId, ['case_id,stage,run_mode,normalization_mode,height_target_m,' ...
    'pattern_id,pattern_name,replicate,source_fit_run,' ...
    'clumped_temperature_file,topography_file,status\n']);
fprintf(qcId, ['case_id,max_elevation_m,mean_elevation_m,std_elevation_m,' ...
    'positive_area_fraction,relief_m,autocorrelation_50km,finite_grid,' ...
    'target_height_error_m,status\n']);

for iMode = 1:numel(normalizationModes)
    mode = normalizationModes(iMode);
    for iHeight = 1:numel(heightLevels)
        targetHeight = heightLevels(iHeight);
        for iPattern = 1:numel(patterns)
            pattern = patterns(iPattern);
            caseId = sprintf('%s_H%d_%s', char(mode), targetHeight, ...
                char(pattern.id));
            caseDir = fullfile(calcRoot, caseId);
            if isfolder(caseDir)
                error('Case directory already exists; refusing to overwrite: %s', ...
                    caseDir);
            end
            mkdir(caseDir);
            mkdir(fullfile(caseDir, 'proxy_clumped'));

            hGrid = rescaleTopography(pattern.hGrid, targetHeight);
            topoFile = fullfile(caseDir, 'Tibet_Eocene_30Ma_topo.mat');
            lon = base.lon; %#ok<NASGU>
            lat = base.lat; %#ok<NASGU>
            save(topoFile, 'lon', 'lat', 'hGrid', '-v7.3');
            copyfile(sampleFile, fullfile(caseDir, ...
                'Tibet_Eocene_30Ma_samples.xlsx'));
            copyfile(divideFile, fullfile(caseDir, ...
                'Tibet_Eocene_30Ma_topo_divide_main.mat'));
            copyfile(clumpedFile, fullfile(caseDir, 'proxy_clumped', ...
                'clumped_temperature.xlsx'));

            runName = ['Tibet_Eocene_30Ma_OxygenClumped_TopoExp_', ...
                caseId, '.run'];
            bestRunName = ['Tibet_Eocene_30Ma_OxygenClumped_TopoExp_', ...
                caseId, '_Best.run'];
            writeCaseRun(clumpedRun, fullfile(caseDir, runName), caseDir, ...
                caseId, false);
            writeCaseRun(clumpedBestRun, fullfile(caseDir, bestRunName), ...
                caseDir, caseId, true);
            if mode == "Nmfixed"
                adjustBestRunM(fullfile(caseDir, bestRunName), ...
                    max(base.hGrid(:)), targetHeight);
            end
            writeCaseMetadata(caseDir, caseId, mode, targetHeight, pattern, ...
                clumpedRun, clumpedFile);
            writeCaseReadme(caseDir, caseId, mode, targetHeight, pattern, ...
                runName, bestRunName);

            [maxH, meanH, stdH, areaFrac, relief, ac50] = ...
                topographyMetrics(hGrid, base.lon, base.lat);
            fprintf(manifestId, '%s,calc_only,calc_only,%s,%d,%s,%s,0,%s,%s,%s,generated\n', ...
                caseId, mode, targetHeight, pattern.id, pattern.name, ...
                'Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive.run', ...
                'proxy_clumped/clumped_temperature.xlsx', ...
                'Tibet_Eocene_30Ma_topo.mat');
            fprintf(qcId, '%s,%.6f,%.6f,%.6f,%.9f,%.6f,%.9f,1,%.9f,pass\n', ...
                caseId, maxH, meanH, stdH, areaFrac, relief, ac50, ...
                maxH - targetHeight);
        end
    end
end

fprintf('Created %d topography sensitivity case skeletons under:\n%s\n', ...
    numel(normalizationModes) * numel(heightLevels) * numel(patterns), calcRoot);
end

function patterns = buildPatterns(base, original)
patterns = struct('id', {}, 'name', {}, 'hGrid', {});
patterns(1) = struct('id', "P01_double_platform", ...
    'name', "current north plus south platform", 'hGrid', base.hGrid);
patterns(2) = struct('id', "P02_north_platform", ...
    'name', "north platform with south belt restored", ...
    'hGrid', applyPlatformBands(original.hGrid, original.lat, true, false));
patterns(3) = struct('id', "P03_south_platform", ...
    'name', "south platform with north belt restored", ...
    'hGrid', applyPlatformBands(original.hGrid, original.lat, false, true));
patterns(4) = struct('id', "P04_broad_plateau", ...
    'name', "broad east-west belt with lower relief", ...
    'hGrid', buildBroadPlateau(original.lat, original.hGrid));
end

function hGrid = applyPlatformBands(hGrid, lat, useNorth, useSouth)
if useSouth
    hGrid = applyPlateauBand(hGrid, bandWeight(lat, 29.15, 0.16, 0.18), ...
        2350, 2900);
end
if useNorth
    hGrid = applyPlateauBand(hGrid, bandWeight(lat, 33.20, 0.22, 0.20), ...
        3200, 4150);
end
hGrid = max(hGrid, 0);
end

function hGrid = buildBroadPlateau(lat, hGrid)
% P04 is an east-west belt: its modification is constant along longitude.
% The latitude-only taper avoids introducing a circular or localized lobe.
weight = bandWeight(lat, 31.3, 1.2, 1.2);
target = 3400;
for i = 1:size(hGrid, 1)
    hGrid(i, :) = hGrid(i, :) + weight(i) .* max(target - hGrid(i, :), 0);
end
hGrid = min(hGrid, 4150);
hGrid = max(hGrid, 0);
end

function hGridOut = rescaleTopography(hGrid, targetHeight)
hGrid = max(hGrid, 0);
hGridOut = hGrid .* (targetHeight / max(hGrid(:)));
end

function writeCaseRun(sourceRun, targetRun, caseDir, caseId, isBest)
lines = readlines(sourceRun, 'WhitespaceRule', 'preserve');
idx = findActiveLineIndices(lines);
if isBest
    suffix = ' fixed best-fit calculation';
else
    suffix = ' clumped fit template';
end
lines(idx(1)) = "Qiangtang topography sensitivity " + caseId + suffix;
lines(idx(3)) = string(caseDir);
writelines(lines, targetRun);
end

function adjustBestRunM(bestRunFile, sourceHeight, targetHeight)
lines = readlines(bestRunFile, 'WhitespaceRule', 'preserve');
idx = findActiveLineIndices(lines);
beta = str2num(lines(idx(end))); %#ok<ST2NM>
if numel(beta) ~= 19
    error('Expected 19 beta values in best run: %s', bestRunFile);
end
scale = targetHeight / sourceHeight;
beta([4, 14]) = beta([4, 14]) .* scale;
lines(idx(end)) = sprintf('%.8g\t', beta);
writelines(lines, bestRunFile);
end

function writeCaseMetadata(caseDir, caseId, mode, targetHeight, pattern, ...
    sourceRun, clumpedFile)
fid = fopen(fullfile(caseDir, 'case_metadata.csv'), 'w');
if fid == -1
    error('Could not create case metadata: %s', caseDir);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'field,value\n');
fprintf(fid, 'case_id,%s\n', caseId);
fprintf(fid, 'stage,calc_only\n');
fprintf(fid, 'run_mode,calc_only\n');
fprintf(fid, 'normalization_mode,%s\n', mode);
fprintf(fid, 'height_target_m,%g\n', targetHeight);
fprintf(fid, 'pattern_id,%s\n', pattern.id);
fprintf(fid, 'pattern_name,"%s"\n', pattern.name);
fprintf(fid, 'source_clumped_run,%s\n', sourceRun);
fprintf(fid, 'clumped_temperature_file,%s\n', clumpedFile);
fprintf(fid, 'spatial_geometry,east_west_band_latitude_only\n');
fprintf(fid, 'divide_interpretation,not_used_as_causal_factor\n');
end

function writeCaseReadme(caseDir, caseId, mode, targetHeight, pattern, ...
    runName, bestRunName)
fid = fopen(fullfile(caseDir, 'README.txt'), 'w');
if fid == -1
    error('Could not create case README: %s', caseDir);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'Qiangtang topography sensitivity case\n');
fprintf(fid, 'Case ID: %s\n', caseId);
fprintf(fid, 'Stage: calc_only\n');
fprintf(fid, 'Normalization: %s\n', mode);
fprintf(fid, 'Target maximum elevation: %.1f m\n', targetHeight);
fprintf(fid, 'Pattern: %s (%s)\n', pattern.id, pattern.name);
fprintf(fid, 'Primary clumped run: %s\n', runName);
fprintf(fid, 'Fixed best-fit clumped run: %s\n', bestRunName);
fprintf(fid, 'Clumped input: proxy_clumped/clumped_temperature.xlsx\n');
fprintf(fid, 'Spatial geometry: east-west bands using latitude-only weights.\n');
fprintf(fid, 'No fit or OPI calculation has been run by the setup function.\n');
end

function [maxH, meanH, stdH, areaFrac, relief, ac50] = ...
    topographyMetrics(hGrid, lon, lat)
finite = isfinite(hGrid);
maxH = max(hGrid(finite));
meanH = mean(hGrid(finite));
stdH = std(hGrid(finite), 1);
areaFrac = mean(hGrid(finite) > 0);
relief = maxH - min(hGrid(finite));
dx = mean(diff(lon)) * 111320 * cosd(mean(lat));
dy = mean(diff(lat)) * 110540;
sx = max(1, round(50000 / abs(dx)));
sy = max(1, round(50000 / abs(dy)));
a = hGrid(1:end-sy, 1:end-sx);
b = hGrid(1+sy:end, 1+sx:end);
valid = isfinite(a) & isfinite(b);
if nnz(valid) < 3
    ac50 = nan;
else
    c = corrcoef(a(valid), b(valid));
    ac50 = c(1, 2);
end
end

function weights = bandWeight(lat, center, halfWidth, transition)
distance = abs(lat(:) - center);
weights = zeros(size(distance));
weights(distance <= halfWidth) = 1;
mask = distance > halfWidth & distance < halfWidth + transition;
weights(mask) = 1 - smoothstep((distance(mask) - halfWidth) ./ transition);
end

function hGridOut = applyPlateauBand(hGridIn, rowWeights, shoulder, plateau)
hGridOut = hGridIn;
for i = 1:size(hGridIn, 1)
    w = rowWeights(i);
    if w <= 0
        continue
    end
    row = hGridIn(i, :);
    t = smoothstep((row - shoulder) ./ max(plateau - shoulder, eps));
    rowOut = row + w .* (plateau - row) .* t;
    rowOut = min(rowOut, plateau);
    hGridOut(i, :) = rowOut;
end
end

function y = smoothstep(x)
x = min(max(x, 0), 1);
y = x .* x .* (3 - 2 .* x);
end

function idx = findActiveLineIndices(lines)
idx = [];
for i = 1:numel(lines)
    str = strip(string(lines(i)));
    if strlength(str) == 0 || startsWith(str, "%")
        continue
    end
    idx(end+1) = i; %#ok<AGROW>
end
if numel(idx) < 10
    error('Run file format does not contain the expected active lines.');
end
end

function closeFiles(manifestId, qcId)
if manifestId ~= -1
    fclose(manifestId);
end
if qcId ~= -1
    fclose(qcId);
end
end
