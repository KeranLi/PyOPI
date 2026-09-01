function opiSetup_TopographyRefitCases(rootScenario, experimentName)
% Create selected joint oxygen plus clumped-temperature refit cases.
% This function only copies inputs and writes run files; it never fits.

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
calcRoot = fullfile(experimentRoot, 'calc_only');
refitRoot = fullfile(experimentRoot, 'refit_selected');
if ~isfolder(refitRoot)
    mkdir(refitRoot);
end

selected = [
    "Mfixed_H3000_P01_double_platform"
    "Mfixed_H5000_P01_double_platform"
    "Mfixed_H3000_P03_south_platform"
    "Mfixed_H5000_P03_south_platform"
    "Nmfixed_H3000_P03_south_platform"
    "Nmfixed_H5000_P04_broad_plateau"
    "Mfixed_H4150_P02_north_platform"
    ];

manifest = fullfile(refitRoot, 'refit_manifest.csv');
fid = fopen(manifest, 'w');
if fid == -1
    error('Could not create refit manifest: %s', manifest);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, ['case_id,source_calc_case,replicate,normalization_mode,' ...
    'height_target_m,pattern_id,run_mode,status\n']);

for i = 1:numel(selected)
    sourceCase = selected(i);
    sourceDir = fullfile(calcRoot, sourceCase);
    if ~isfolder(sourceDir)
        error('Selected calc-only case not found: %s', sourceDir);
    end
    targetCase = sourceCase + "_R01";
    targetDir = fullfile(refitRoot, targetCase);
    if isfolder(targetDir)
        existingRuns = dir(fullfile(targetDir, '*.run'));
        if ~isempty(existingRuns)
            error('Refit case already exists; refusing to overwrite: %s', targetDir);
        end
        warning('Continuing an incomplete refit case directory: %s', targetDir);
    else
        mkdir(targetDir);
        mkdir(fullfile(targetDir, 'proxy_clumped'));
    end
    if ~isfolder(fullfile(targetDir, 'proxy_clumped'))
        mkdir(fullfile(targetDir, 'proxy_clumped'));
    end

    copyfile(fullfile(sourceDir, 'Tibet_Eocene_30Ma_topo.mat'), ...
        fullfile(targetDir, 'Tibet_Eocene_30Ma_topo.mat'));
    copyfile(fullfile(sourceDir, 'Tibet_Eocene_30Ma_samples.xlsx'), ...
        fullfile(targetDir, 'Tibet_Eocene_30Ma_samples.xlsx'));
    copyfile(fullfile(sourceDir, 'Tibet_Eocene_30Ma_topo_divide_main.mat'), ...
        fullfile(targetDir, 'Tibet_Eocene_30Ma_topo_divide_main.mat'));
    copyfile(fullfile(sourceDir, 'proxy_clumped', ...
        'clumped_temperature.xlsx'), fullfile(targetDir, 'proxy_clumped', ...
        'clumped_temperature.xlsx'));

    primary = findPrimaryRun(sourceDir);
    runName = "Tibet_Eocene_30Ma_OxygenClumped_TopoExp_" + targetCase + ".run";
    writeRefitRun(primary, fullfile(targetDir, runName), targetDir, targetCase);
    [mode, targetHeight, pattern] = parseCaseId(sourceCase);
    writeCaseMetadata(targetDir, targetCase, sourceCase, mode, ...
        targetHeight, pattern, runName);
    fprintf(fid, '%s,%s,R01,%s,%g,%s,refit,generated\n', targetCase, ...
        sourceCase, mode, targetHeight, pattern);
end

fprintf('Created %d selected clumped refit case skeletons under:\n%s\n', ...
    numel(selected), refitRoot);
end

function runFile = findPrimaryRun(caseDir)
files = dir(fullfile(caseDir, '*.run'));
files = files(~endsWith(string({files.name}), "_Best.run"));
if numel(files) ~= 1
    error('Expected one primary run in %s, found %d.', caseDir, numel(files));
end
runFile = fullfile(files(1).folder, files(1).name);
end

function writeRefitRun(sourceRun, targetRun, targetDir, caseId)
lines = readlines(sourceRun, 'WhitespaceRule', 'preserve');
idx = findActiveLineIndices(lines);
lines(idx(1)) = "Qiangtang topography joint clumped refit " + caseId;
lines(idx(3)) = string(targetDir);
    targetRun = string(targetRun);
    if ~isscalar(targetRun)
        error('Refit run path must be a scalar path: %s', targetRun);
    end
    writelines(lines, targetRun);
end

function [mode, targetHeight, pattern] = parseCaseId(caseId)
parts = split(string(caseId), '_');
mode = parts(1);
targetHeight = str2double(extractAfter(parts(2), 1));
pattern = join(parts(3:end), '_');
end

function writeCaseMetadata(caseDir, caseId, sourceCase, mode, targetHeight, ...
    pattern, runName)
fid = fopen(fullfile(caseDir, 'case_metadata.csv'), 'w');
if fid == -1
    error('Could not create metadata in %s', caseDir);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'field,value\n');
fprintf(fid, 'case_id,%s\n', caseId);
fprintf(fid, 'source_calc_case,%s\n', sourceCase);
fprintf(fid, 'stage,refit_selected\n');
fprintf(fid, 'run_mode,refit\n');
fprintf(fid, 'replicate,R01\n');
fprintf(fid, 'normalization_mode,%s\n', mode);
fprintf(fid, 'height_target_m,%g\n', targetHeight);
fprintf(fid, 'pattern_id,%s\n', pattern);
fprintf(fid, 'fit_workflow,oxygen_plus_clumped_temperature\n');
fprintf(fid, 'primary_run,%s\n', runName);
fprintf(fid, 'clumped_input,proxy_clumped/clumped_temperature.xlsx\n');
fprintf(fid, 'spatial_geometry,east_west_band_latitude_only\n');
fprintf(fid, 'divide_interpretation,not_used_as_causal_factor\n');
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
    error('Run file format does not contain expected active lines.');
end
end
