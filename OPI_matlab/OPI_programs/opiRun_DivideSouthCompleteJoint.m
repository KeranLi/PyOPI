function opiRun_DivideSouthCompleteJoint()
% Run fixed-beta southern-Gangdese divide simulations for both topographies.
%
% The beta vector is taken from each complete joint oxygen+clumped fit.  The
% divide geometry is changed independently, then OPI calculation and maps are
% regenerated in new directories.  Original baseline files are untouched.

programRoot = fileparts(mfilename('fullpath'));
scenarioRoot = fullfile(fileparts(programRoot), 'scenarios', ...
    'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
completeRoot = fullfile(scenarioRoot, 'complete_joint_simulation');
divideRoot = scenarioRoot;

scenarios = [
    struct('name', 'Q4_V35_G4')
    struct('name', 'Q4_V25_G4')];
shifts = [
    struct('name', 'south_010deg', 'file', 'Tibet_Eocene_30Ma_topo_divide_main_south010.mat')
    struct('name', 'south_020deg', 'file', 'Tibet_Eocene_30Ma_topo_divide_main_south020.mat')
    struct('name', 'south_030deg', 'file', 'Tibet_Eocene_30Ma_topo_divide_main_south030.mat')];

outputRootName = 'divide_south_experiments_fixed_joint';
for i = 1:numel(scenarios)
    sourceDir = fullfile(completeRoot, scenarios(i).name);
    baselineRun = fullfile(sourceDir, ...
        ['Tibet_Eocene_30Ma_', scenarios(i).name, '_Best.run']);
    if ~isfile(baselineRun)
        error('Baseline best run is missing: %s', baselineRun);
    end

    for j = 1:numel(shifts)
        caseDir = fullfile(sourceDir, outputRootName, shifts(j).name);
        if ~exist(caseDir, 'dir')
            mkdir(caseDir);
        end
        prepareCaseInputs(sourceDir, caseDir, divideRoot, shifts(j).file);
        runFile = writeDivideBestRun(baselineRun, caseDir, scenarios(i).name, shifts(j));

        fprintf('\n===== %s / %s =====\n', scenarios(i).name, shifts(j).name);
        fprintf('Run file: %s\n', runFile);
        opiCalc_TwoWinds_OxygenOnly(runFile);
        resultFile = fullfile(caseDir, 'opiCalc_TwoWinds_OxygenOnly_Results.mat');
        opiMaps_TwoWinds(resultFile);
        writeCaseReadme(caseDir, scenarios(i).name, shifts(j));
    end
end

writeSummary(completeRoot, scenarios, shifts, outputRootName);
fprintf('\nCompleted fixed-beta southern-divide simulations.\n');
end

function prepareCaseInputs(sourceDir, caseDir, divideRoot, divideFile)
copyIfMissing(fullfile(sourceDir, 'Tibet_Eocene_30Ma_topo.mat'), caseDir);
copyIfMissing(fullfile(sourceDir, 'Tibet_Eocene_30Ma_samples.xlsx'), caseDir);
copyIfMissing(fullfile(sourceDir, 'collected_proxy_observations.csv'), caseDir);
copyIfMissing(fullfile(sourceDir, 'collected_carbonate_assimilation_config.csv'), caseDir);
proxyDir = fullfile(sourceDir, 'proxy_clumped');
if isfolder(proxyDir) && ~isfolder(fullfile(caseDir, 'proxy_clumped'))
    copyfile(proxyDir, fullfile(caseDir, 'proxy_clumped'));
end
sourceDivide = fullfile(divideRoot, divideFile);
if ~isfile(sourceDivide)
    error('Southern divide file is missing: %s', sourceDivide);
end
targetDivide = fullfile(caseDir, divideFile);
if ~isfile(targetDivide)
    copyfile(sourceDivide, targetDivide);
end
end

function runFile = writeDivideBestRun(baselineRun, caseDir, scenarioName, shift)
text = fileread(baselineRun);
sourceDir = fileparts(baselineRun);
text = strrep(text, sourceDir, caseDir);
text = strrep(text, 'Tibet_Eocene_30Ma_topo_divide_main.mat', shift.file);
text = strrep(text, 'complete_joint_simulation', 'divide_south_experiments_fixed_joint');
text = sprintf('%% Southern Gangdese divide fixed-beta simulation: %s\n%% Divide shift: %s\n%s', ...
    scenarioName, shift.name, text);
runFile = fullfile(caseDir, sprintf('Tibet_Eocene_30Ma_%s_%s_Best.run', ...
    scenarioName, shift.name));
fid = fopen(runFile, 'w', 'native', 'UTF-8');
if fid == -1
    error('Could not create run file: %s', runFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s', text);
end

function copyIfMissing(sourceFile, targetDir)
targetFile = fullfile(targetDir, fileparts(sourceFile)); %#ok<NASGU>
[~, name, ext] = fileparts(sourceFile);
targetFile = fullfile(targetDir, [name, ext]);
if ~isfile(targetFile)
    copyfile(sourceFile, targetFile);
end
end

function writeCaseReadme(caseDir, scenarioName, shift)
readme = fullfile(caseDir, 'README.txt');
fid = fopen(readme, 'w', 'native', 'UTF-8');
if fid == -1
    error('Could not create README: %s', readme);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'Scenario: %s\n', scenarioName);
fprintf(fid, 'Southern Gangdese divide: %s\n', shift.name);
fprintf(fid, 'Mode: fixed beta from the corresponding collected/paleosol joint fit.\n');
fprintf(fid, 'The divide geometry is changed only for opiCalc and opiMaps.\n');
fprintf(fid, 'Original complete_joint_simulation files are unchanged.\n');
end

function writeSummary(completeRoot, scenarios, shifts, outputRootName)
rows = cell(numel(scenarios) * numel(shifts), 8);
k = 0;
for i = 1:numel(scenarios)
    for j = 1:numel(shifts)
        k = k + 1;
        caseDir = fullfile(completeRoot, scenarios(i).name, outputRootName, shifts(j).name);
        resultFile = fullfile(caseDir, 'opiCalc_TwoWinds_OxygenOnly_Results.mat');
        rows(k, :) = {scenarios(i).name, shifts(j).name, resultFile, ...
            fullfile(caseDir, 'opiMaps_TwoWinds_Fig01.pdf'), ...
            isfile(resultFile), isfile(fullfile(caseDir, 'opiMaps_TwoWinds_Fig33.pdf')), ...
            NaN, NaN};
        if isfile(resultFile)
            S = load(resultFile, 'chiR2', 'nu');
            rows{k, 7} = S.chiR2;
            rows{k, 8} = S.nu;
        end
    end
end
T = cell2table(rows, 'VariableNames', {'scenario','divide_shift','result_file', ...
    'fig01_file','has_result','has_fig33','oxygen_chiR2','oxygen_nu'});
writetable(T, fullfile(completeRoot, outputRootName, 'divide_south_fixed_joint_summary.csv'));
end
