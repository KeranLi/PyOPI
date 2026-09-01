function opiSetup_ValleyHeightSensitivity(rootScenario, experimentName)
% Create fixed-parameter cases that isolate intermontane valley elevation.

if nargin < 1 || strlength(string(rootScenario)) == 0
    rootScenario = fullfile(fileparts(mfilename('fullpath')), '..', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
end
if nargin < 2 || strlength(string(experimentName)) == 0
    experimentName = 'topography_valley_height_sensitivity';
end
rootScenario = char(string(rootScenario));
experimentRoot = fullfile(rootScenario, char(string(experimentName)));
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

cases = makeCases();
manifestId = fopen(fullfile(designRoot, 'case_manifest.csv'), 'w');
qcId = fopen(fullfile(designRoot, 'topography_quality_control.csv'), 'w');
if manifestId == -1 || qcId == -1
    error('Could not open valley experiment design tables.');
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
    hGrid = opiBuild_EastWestBandTopography(base.hGrid, base.lat, ...
        c.gangdese, c.qiangtang, c.valleyMode, c.valley);
    lon = base.lon; %#ok<NASGU>
    lat = base.lat; %#ok<NASGU>
    save(fullfile(caseDir, 'Tibet_Eocene_30Ma_topo.mat'), ...
        'lon', 'lat', 'hGrid', '-v7.3');
    copyfile(sampleFile, fullfile(caseDir, 'Tibet_Eocene_30Ma_samples.xlsx'));
    copyfile(divideFile, fullfile(caseDir, ...
        'Tibet_Eocene_30Ma_topo_divide_main.mat'));
    copyfile(clumpedFile, fullfile(caseDir, 'proxy_clumped', ...
        'clumped_temperature.xlsx'));

    runName = ['Tibet_Eocene_30Ma_OxygenClumped_Valley_', c.id, '.run'];
    bestName = ['Tibet_Eocene_30Ma_OxygenClumped_Valley_', c.id, '_Best.run'];
    writeRun(sourceRun, fullfile(caseDir, runName), caseDir, c.id, false);
    writeRun(sourceBest, fullfile(caseDir, bestName), caseDir, c.id, true);
    writeMetadata(caseDir, c, runName, bestName);

    gMean = coreMean(hGrid, base.lat, 29.15, 0.35);
    qMean = coreMean(hGrid, base.lat, 33.20, 0.45);
    vMean = coreMean(hGrid, base.lat, 31.30, 0.45);
    pass = abs(gMean - c.gangdese) <= 75 && abs(qMean - c.qiangtang) <= 75;
    if c.valleyMode ~= "Vnone"
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
writeReadme(experimentRoot, numel(cases));
fprintf('Created %d valley-height sensitivity cases under:\n%s\n', ...
    numel(cases), calcRoot);
end

function cases = makeCases()
valleyValues = 1500:500:3000;
cases = struct('id', {}, 'gangdese', {}, 'qiangtang', {}, ...
    'valleyMode', {}, 'valley', {});
k = 0;
for g = [3000, 4500]
    for v = valleyValues
        k = k + 1;
        cases(k) = struct('id', sprintf('G%d_Q4000_V%d', g, v), ...
            'gangdese', g, 'qiangtang', 4000, ...
            'valleyMode', "V" + v, 'valley', v);
    end
    k = k + 1;
    cases(k) = struct('id', sprintf('G%d_Q4000_Vnone', g), ...
        'gangdese', g, 'qiangtang', 4000, ...
        'valleyMode', "Vnone", 'valley', nan);
end
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
lines(idx(1)) = "Valley height sensitivity " + caseId + suffix;
lines(idx(3)) = string(caseDir);
writelines(lines, targetFile);
end

function writeMetadata(caseDir, c, runName, bestName)
fid = fopen(fullfile(caseDir, 'case_metadata.csv'), 'w');
if fid == -1
    error('Could not write valley case metadata.');
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
end

function writeReadme(experimentRoot, count)
fid = fopen(fullfile(experimentRoot, 'README.md'), 'w');
if fid == -1
    error('Could not write valley experiment README.');
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Intermontane Valley Height Sensitivity\n\n');
fprintf(fid, 'Cases generated: **%d**. Qiangtang is fixed at 4000 m; Gangdese is 3000 or 4500 m; imposed valley floors range from 1500 to 3000 m, plus a continuous-highland control.\n\n', count);
fprintf(fid, 'The primary response is 50 km precipitation-weighted d18O relative to -13.54 per mil.\n');
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
