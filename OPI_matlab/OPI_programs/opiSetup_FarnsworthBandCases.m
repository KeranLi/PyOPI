function opiSetup_FarnsworthBandCases(rootScenario, experimentName)
% Create fixed-parameter east-west-band cases inspired by Farnsworth (2021).
% This function writes case inputs and quality-control tables; it runs no OPI.

if nargin < 1 || strlength(string(rootScenario)) == 0
    rootScenario = fullfile(fileparts(mfilename('fullpath')), '..', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
end
if nargin < 2 || strlength(string(experimentName)) == 0
    experimentName = 'topography_farnsworth_band';
end
rootScenario = char(string(rootScenario));
experimentName = char(string(experimentName));

experimentRoot = fullfile(rootScenario, experimentName);
designRoot = fullfile(experimentRoot, 'design');
calcRoot = fullfile(experimentRoot, 'calc_only');
makeDir(experimentRoot);
makeDir(designRoot);
makeDir(calcRoot);

sourceRun = fullfile(rootScenario, 'oxygen_clumped_ultra_aggressive', ...
    'Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive.run');
sourceBestRun = fullfile(rootScenario, 'oxygen_clumped_ultra_aggressive', ...
    'Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive_Best.run');
sampleFile = fullfile(rootScenario, 'Tibet_Eocene_30Ma_samples.xlsx');
clumpedFile = fullfile(rootScenario, 'proxy_clumped', ...
    'clumped_temperature.xlsx');
divideFile = fullfile(rootScenario, ...
    'Tibet_Eocene_30Ma_topo_divide_main.mat');
topoFile = fullfile(rootScenario, 'Tibet_Eocene_30Ma_topo.mat');
required = string({sourceRun, sourceBestRun, sampleFile, clumpedFile, ...
    divideFile, topoFile});
for i = 1:numel(required)
    if ~isfile(required(i))
        error('Required baseline file not found: %s', required(i));
    end
end

base = load(topoFile, 'lon', 'lat', 'hGrid');
cases = experimentCases();

manifestFile = fullfile(designRoot, 'case_manifest.csv');
qcFile = fullfile(designRoot, 'topography_quality_control.csv');
manifestId = fopen(manifestFile, 'w');
qcId = fopen(qcFile, 'w');
if manifestId == -1 || qcId == -1
    error('Could not create experiment design tables.');
end
cleanup = onCleanup(@() closeFiles(manifestId, qcId)); %#ok<NASGU>

fprintf(manifestId, ['case_id,stage,gangdese_target_m,qiangtang_target_m,' ...
    'valley_mode,valley_target_m,parameter_mode,spatial_geometry,status\n']);
fprintf(qcId, ['case_id,max_elevation_m,gangdese_core_mean_m,' ...
    'qiangtang_core_mean_m,valley_core_mean_m,finite_grid,' ...
    'longitude_localized_mask,target_check,status\n']);

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
    copyfile(sampleFile, fullfile(caseDir, ...
        'Tibet_Eocene_30Ma_samples.xlsx'));
    copyfile(divideFile, fullfile(caseDir, ...
        'Tibet_Eocene_30Ma_topo_divide_main.mat'));
    copyfile(clumpedFile, fullfile(caseDir, 'proxy_clumped', ...
        'clumped_temperature.xlsx'));

    primaryName = ['Tibet_Eocene_30Ma_OxygenClumped_Farnsworth_', ...
        c.id, '.run'];
    bestName = ['Tibet_Eocene_30Ma_OxygenClumped_Farnsworth_', ...
        c.id, '_Best.run'];
    writeCaseRun(sourceRun, fullfile(caseDir, primaryName), caseDir, c.id, false);
    writeCaseRun(sourceBestRun, fullfile(caseDir, bestName), caseDir, c.id, true);
    writeMetadata(caseDir, c, primaryName, bestName, sourceRun);
    writeReadme(caseDir, c, primaryName, bestName);

    gMean = coreMean(hGrid, base.lat, 29.15, 0.35);
    qMean = coreMean(hGrid, base.lat, 33.20, 0.45);
    vMean = coreMean(hGrid, base.lat, 31.30, 0.45);
    targetPass = abs(gMean - c.gangdese) <= 75 && ...
        abs(qMean - c.qiangtang) <= 75;
    if c.valleyMode == "V1500"
        targetPass = targetPass && abs(vMean - c.valley) <= 75;
    end
    status = "pass";
    if ~targetPass
        status = "review";
    end
    fprintf(manifestId, '%s,%d,%g,%g,%s,%g,fixed_clumped_best,east_west_latitude_only,generated\n', ...
        c.id, c.stage, c.gangdese, c.qiangtang, c.valleyMode, ...
        c.valley);
    fprintf(qcId, '%s,%.6f,%.6f,%.6f,%.6f,%d,0,%d,%s\n', ...
        c.id, max(hGrid(:)), gMean, qMean, vMean, ...
        all(isfinite(hGrid(:))), targetPass, status);
end

writeDesignReadme(experimentRoot, numel(cases));
fprintf('Created %d Farnsworth-inspired band cases under:\n%s\n', ...
    numel(cases), calcRoot);
fprintf('Quality control table:\n%s\n', qcFile);
end

function cases = experimentCases()
spec = {
    'G3000_Q3000_Vnone', 1, 3000, 3000, "Vnone", nan
    'G3000_Q3500_Vnone', 1, 3000, 3500, "Vnone", nan
    'G3000_Q4000_Vnone', 1, 3000, 4000, "Vnone", nan
    'G3000_Q4500_Vnone', 1, 3000, 4500, "Vnone", nan
    'G3000_Q5000_Vnone', 1, 3000, 5000, "Vnone", nan
    'G4500_Q4000_V1500', 2, 4500, 4000, "V1500", 1500
    'G4500_Q4000_Vnone', 2, 4500, 4000, "Vnone", nan
    'G1500_Q4000_Vnone', 2, 1500, 4000, "Vnone", nan
    'G2000_Q2000_Vnone', 2, 2000, 2000, "Vnone", nan
    };
cases = struct('id', {}, 'stage', {}, 'gangdese', {}, 'qiangtang', {}, ...
    'valleyMode', {}, 'valley', {});
for i = 1:size(spec, 1)
    cases(i) = struct('id', spec{i, 1}, 'stage', spec{i, 2}, ...
        'gangdese', spec{i, 3}, 'qiangtang', spec{i, 4}, ...
        'valleyMode', spec{i, 5}, 'valley', spec{i, 6});
end
end

function hGrid = buildTopography(hGrid, lat, c)
% Core rows are flat east-west targets. Longitudinal relief from the source
% topography is retained only through the smooth transition rows.
hGrid = imposeBand(hGrid, lat, 29.15, 0.35, 0.45, c.gangdese);
hGrid = imposeBand(hGrid, lat, 33.20, 0.45, 0.45, c.qiangtang);
if c.valleyMode == "V1500"
    hGrid = imposeBand(hGrid, lat, 31.30, 0.45, 0.55, c.valley);
else
    hGrid = imposeIntermontaneRamp(hGrid, lat, c.gangdese, c.qiangtang);
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

function hGrid = imposeIntermontaneRamp(hGrid, lat, gangdese, qiangtang)
center = 31.30;
halfWidth = 0.45;
transition = 0.55;
w = bandWeight(lat, center, halfWidth, transition);
south = center - halfWidth - transition;
north = center + halfWidth + transition;
for i = 1:size(hGrid, 1)
    if w(i) > 0
        f = min(max((lat(i) - south) / (north - south), 0), 1);
        target = gangdese + f * (qiangtang - gangdese);
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
values = hGrid(mask, :);
value = mean(values(:), 'omitnan');
end

function writeCaseRun(sourceFile, targetFile, caseDir, caseId, isBest)
lines = readlines(sourceFile, 'WhitespaceRule', 'preserve');
idx = activeLines(lines);
if isBest
    suffix = ' fixed best-fit calculation';
else
    suffix = ' joint oxygen plus clumped refit template';
end
lines(idx(1)) = "Farnsworth-inspired topography " + caseId + suffix;
lines(idx(3)) = string(caseDir);
writelines(lines, targetFile);
end

function writeMetadata(caseDir, c, primaryName, bestName, sourceRun)
fid = fopen(fullfile(caseDir, 'case_metadata.csv'), 'w');
if fid == -1
    error('Could not write metadata in %s.', caseDir);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'field,value\n');
fprintf(fid, 'case_id,%s\n', c.id);
fprintf(fid, 'stage,%d\n', c.stage);
fprintf(fid, 'gangdese_target_m,%g\n', c.gangdese);
fprintf(fid, 'qiangtang_target_m,%g\n', c.qiangtang);
fprintf(fid, 'valley_mode,%s\n', c.valleyMode);
fprintf(fid, 'valley_target_m,%g\n', c.valley);
fprintf(fid, 'parameter_mode,fixed_clumped_best\n');
fprintf(fid, 'spatial_geometry,east_west_latitude_only\n');
fprintf(fid, 'primary_refit_template,%s\n', primaryName);
fprintf(fid, 'fixed_calculation_run,%s\n', bestName);
fprintf(fid, 'source_clumped_run,%s\n', sourceRun);
fprintf(fid, 'primary_comparison,reconstructed_precipitation_d18O\n');
fprintf(fid, 'divide_interpretation,not_used_as_causal_factor\n');
end

function writeReadme(caseDir, c, primaryName, bestName)
fid = fopen(fullfile(caseDir, 'README.txt'), 'w');
if fid == -1
    error('Could not write README in %s.', caseDir);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'Farnsworth-inspired east-west-band topography case\n');
fprintf(fid, 'Case ID: %s\n', c.id);
fprintf(fid, 'Gangdese target: %.0f m\n', c.gangdese);
fprintf(fid, 'Qiangtang target: %.0f m\n', c.qiangtang);
fprintf(fid, 'Valley mode: %s\n', c.valleyMode);
fprintf(fid, 'Primary refit template: %s\n', primaryName);
fprintf(fid, 'Fixed-parameter calculation: %s\n', bestName);
fprintf(fid, 'No OPI calculation or fit was run during setup.\n');
end

function writeDesignReadme(experimentRoot, count)
fid = fopen(fullfile(experimentRoot, 'README.md'), 'w');
if fid == -1
    error('Could not write experiment README.');
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Farnsworth-Inspired East-West Band Cases\n\n');
fprintf(fid, 'Generated cases: **%d**.\n\n', count);
fprintf(fid, 'These are fixed-parameter spatial-process experiments. The primary output is reconstructed precipitation d18O.\n\n');
fprintf(fid, 'Run `_Best.run` files only for calc-only experiments. Use the corresponding primary `.run` only for a preregistered oxygen-plus-clumped refit.\n');
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
    error('Run file does not contain the expected active lines.');
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
