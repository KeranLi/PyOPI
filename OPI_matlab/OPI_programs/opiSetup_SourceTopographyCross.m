function opiSetup_SourceTopographyCross(rootScenario, experimentName)
% Create fixed-parameter source-d18O by topography cross experiments.

if nargin < 1 || strlength(string(rootScenario)) == 0
    rootScenario = fullfile(fileparts(mfilename('fullpath')), '..', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
end
if nargin < 2 || strlength(string(experimentName)) == 0
    experimentName = 'source_d18O_topography_cross';
end
rootScenario = char(string(rootScenario));
experimentRoot = fullfile(rootScenario, char(string(experimentName)));
designRoot = fullfile(experimentRoot, 'design');
calcRoot = fullfile(experimentRoot, 'calc_only');
makeDir(designRoot);
makeDir(calcRoot);

fitDir = fullfile(rootScenario, 'oxygen_clumped_ultra_aggressive');
sourceRun = fullfile(fitDir, ...
    'Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive.run');
sourceBest = fullfile(fitDir, ...
    'Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive_Best.run');
sampleFile = fullfile(rootScenario, 'Tibet_Eocene_30Ma_samples.xlsx');
clumpedFile = fullfile(rootScenario, 'proxy_clumped', ...
    'clumped_temperature.xlsx');
divideFile = fullfile(rootScenario, ...
    'Tibet_Eocene_30Ma_topo_divide_main.mat');
required = string({sourceRun, sourceBest, sampleFile, clumpedFile, divideFile});
for i = 1:numel(required)
    if ~isfile(required(i))
        error('Required baseline file not found: %s', required(i));
    end
end

baseBeta = bestBeta(sourceBest);
baseSourcePermil = baseBeta(7) * 1e3;
topographies = topographyCases(rootScenario);
offsets = [-0.5, 0, 0.5];
manifestId = fopen(fullfile(designRoot, 'case_manifest.csv'), 'w');
qcId = fopen(fullfile(designRoot, 'parameter_quality_control.csv'), 'w');
if manifestId == -1 || qcId == -1
    error('Could not open source-topography design tables.');
end
cleanup = onCleanup(@() closeFiles(manifestId, qcId)); %#ok<NASGU>
fprintf(manifestId, ['case_id,topography_id,source_case_dir,' ...
    'd18O0_1_offset_permil,d18O0_1_absolute_permil,parameter_mode,' ...
    'target_d18O_permil,target_sigma_permil,status\n']);
fprintf(qcId, ['case_id,expected_d18O0_1_permil,' ...
    'written_d18O0_1_permil,error_permil,status\n']);

for iTopo = 1:numel(topographies)
    topo = topographies(iTopo);
    sourceTopo = fullfile(topo.sourceDir, 'Tibet_Eocene_30Ma_topo.mat');
    if ~isfile(sourceTopo)
        error('Source topography not found: %s', sourceTopo);
    end
    for iOffset = 1:numel(offsets)
        offset = offsets(iOffset);
        offsetId = offsetToken(offset);
        caseId = sprintf('%s_S1src_%s', topo.id, offsetId);
        caseDir = fullfile(calcRoot, caseId);
        if isfolder(caseDir)
            error('Case already exists; refusing to overwrite: %s', caseDir);
        end
        makeDir(caseDir);
        makeDir(fullfile(caseDir, 'proxy_clumped'));
        copyfile(sourceTopo, fullfile(caseDir, 'Tibet_Eocene_30Ma_topo.mat'));
        copyfile(sampleFile, fullfile(caseDir, ...
            'Tibet_Eocene_30Ma_samples.xlsx'));
        copyfile(divideFile, fullfile(caseDir, ...
            'Tibet_Eocene_30Ma_topo_divide_main.mat'));
        copyfile(clumpedFile, fullfile(caseDir, 'proxy_clumped', ...
            'clumped_temperature.xlsx'));

        runName = ['Tibet_Eocene_30Ma_OxygenClumped_SourceCross_', ...
            caseId, '.run'];
        bestName = ['Tibet_Eocene_30Ma_OxygenClumped_SourceCross_', ...
            caseId, '_Best.run'];
        writeRun(sourceRun, fullfile(caseDir, runName), caseDir, caseId);
        written = writeBestRun(sourceBest, fullfile(caseDir, bestName), ...
            caseDir, caseId, offset);
        expected = baseSourcePermil + offset;
        parameterError = written - expected;
        status = "pass";
        if abs(parameterError) > 1e-6
            status = "fail";
        end
        writeMetadata(caseDir, caseId, topo, offset, expected, ...
            runName, bestName);
        fprintf(manifestId, '%s,%s,%s,%.3f,%.6f,fixed_clumped_best,-13.54,0.5,generated\n', ...
            caseId, topo.id, topo.sourceDir, offset, expected);
        fprintf(qcId, '%s,%.9f,%.9f,%.9f,%s\n', ...
            caseId, expected, written, parameterError, status);
    end
end
writeReadme(experimentRoot, numel(topographies) * numel(offsets), ...
    baseSourcePermil);
fprintf('Created %d source-topography cross cases under:\n%s\n', ...
    numel(topographies) * numel(offsets), calcRoot);
end

function topographies = topographyCases(rootScenario)
heightRoot = fullfile(rootScenario, ...
    'topography_qiangtang_height_refinement', 'calc_only');
valleyRoot = fullfile(rootScenario, ...
    'topography_valley_height_sensitivity', 'calc_only');
spec = {
    'G3000_Q3500_Vnone', heightRoot
    'G3000_Q4000_Vnone', heightRoot
    'G3000_Q4500_Vnone', heightRoot
    'G4500_Q4000_Vnone', heightRoot
    'G3000_Q4000_V1500', valleyRoot
    'G3000_Q4000_V2500', valleyRoot
    };
topographies = struct('id', {}, 'sourceDir', {});
for i = 1:size(spec, 1)
    topographies(i) = struct('id', spec{i, 1}, ...
        'sourceDir', fullfile(spec{i, 2}, spec{i, 1}));
end
end

function token = offsetToken(offset)
if offset < 0
    prefix = 'm';
elseif offset > 0
    prefix = 'p';
else
    token = 'base';
    return
end
token = sprintf('%s%.1f', prefix, abs(offset));
token = strrep(token, '.', 'p');
end

function beta = bestBeta(runFile)
lines = readlines(runFile, 'WhitespaceRule', 'preserve');
idx = activeLines(lines);
beta = str2num(lines(idx(end))); %#ok<ST2NM>
if numel(beta) ~= 19
    error('Expected 19 best-fit beta values in %s.', runFile);
end
end

function writeRun(sourceFile, targetFile, caseDir, caseId)
lines = readlines(sourceFile, 'WhitespaceRule', 'preserve');
idx = activeLines(lines);
lines(idx(1)) = "Source-topography cross " + caseId + ...
    " joint oxygen plus clumped refit template";
lines(idx(3)) = string(caseDir);
writelines(lines, targetFile);
end

function writtenPermil = writeBestRun( ...
    sourceFile, targetFile, caseDir, caseId, offsetPermil)
lines = readlines(sourceFile, 'WhitespaceRule', 'preserve');
idx = activeLines(lines);
beta = str2num(lines(idx(end))); %#ok<ST2NM>
if numel(beta) ~= 19
    error('Expected 19 beta values in %s.', sourceFile);
end
beta(7) = beta(7) + offsetPermil / 1e3;
lines(idx(1)) = "Source-topography cross " + caseId + ...
    " fixed best-fit calculation";
lines(idx(3)) = string(caseDir);
lines(idx(end)) = sprintf('%.12g\t', beta);
writelines(lines, targetFile);
writtenPermil = beta(7) * 1e3;
end

function writeMetadata(caseDir, caseId, topo, offset, absolute, ...
    runName, bestName)
fid = fopen(fullfile(caseDir, 'case_metadata.csv'), 'w');
if fid == -1
    error('Could not write source-topography metadata.');
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'field,value\n');
fprintf(fid, 'case_id,%s\n', caseId);
fprintf(fid, 'topography_id,%s\n', topo.id);
fprintf(fid, 'source_topography_dir,%s\n', topo.sourceDir);
fprintf(fid, 'd18O0_1_offset_permil,%.3f\n', offset);
fprintf(fid, 'd18O0_1_absolute_permil,%.6f\n', absolute);
fprintf(fid, 'all_other_parameters,fixed_clumped_best\n');
fprintf(fid, 'primary_refit_template,%s\n', runName);
fprintf(fid, 'fixed_calculation_run,%s\n', bestName);
fprintf(fid, 'target_d18O_permil,-13.54\n');
fprintf(fid, 'target_sigma_permil,0.5\n');
fprintf(fid, 'primary_spatial_support,50km_precipitation_weighted\n');
end

function writeReadme(experimentRoot, count, baseSourcePermil)
fid = fopen(fullfile(experimentRoot, 'README.md'), 'w');
if fid == -1
    error('Could not write source-topography README.');
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Source d18O by Topography Cross Experiment\n\n');
fprintf(fid, 'Cases generated: **%d**. First-wind source d18O is %.3f per mil at baseline and is perturbed by +/-0.5 per mil.\n\n', count, baseSourcePermil);
fprintf(fid, 'All other parameters use the fixed joint oxygen-plus-clumped best solution. The target is -13.54 +/- 0.5 per mil at 50 km spatial support.\n');
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
