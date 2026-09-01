function opiRun_DivideNorthExperiments()
% opiRun_DivideNorthExperiments runs the Qiangtang divide-north suite.

scenarioRoot = ['/Users/keranli/Desktop/Coding/OPI_matlab/scenarios/', ...
    'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid'];
experimentRoot = fullfile(scenarioRoot, 'divide_north_experiments');
clumpedFile = fullfile(scenarioRoot, 'proxy_clumped', 'clumped_temperature.xlsx');

experiments = [
    struct('folder', 'north_010deg', ...
    'runFile', 'Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive_DivideNorth010.run')
    struct('folder', 'north_020deg', ...
    'runFile', 'Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive_DivideNorth020.run')
    struct('folder', 'north_030deg', ...
    'runFile', 'Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive_DivideNorth030.run')];

for i = 1:numel(experiments)
    runFile = fullfile(experimentRoot, experiments(i).folder, experiments(i).runFile);
    runPath = fileparts(runFile);
    fprintf('\n===== Running divide-north experiment %d/%d =====\n', i, numel(experiments));
    fprintf('%s\n', runFile);

    opiFit_TwoWinds_OxygenClumped(runFile);
    fitFile = fullfile(runPath, 'opiFit_TwoWinds_OxygenClumped_BestFit.mat');
    bestRunFile = writeBestRunFile(runFile, fitFile);

    opiCalc_TwoWinds_OxygenOnly(bestRunFile);
    matFile = fullfile(runPath, 'opiCalc_TwoWinds_OxygenOnly_Results.mat');
    opiCompare_ClumpedTemperature(matFile, clumpedFile, fullfile(runPath, 'proxy_clumped'));
    opiMaps_TwoWinds(matFile);
end

end

function bestRunFile = writeBestRunFile(runFile, fitFile)
S = load(fitFile, 'beta', 'chiR2Total', 'nuTotal', 'detail', ...
    'dolomiteOffsetC', 'sigmaOffsetC', 'clumpedSeason');
[runPath, runName, runExt] = fileparts(runFile);
bestRunFile = fullfile(runPath, [runName, '_Best', runExt]);

text = fileread(runFile);
fid = fopen(bestRunFile, 'w', 'native', 'UTF-8');
if fid == -1
    error('Could not create best-run file: %s', bestRunFile);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', strip(text));
fprintf(fid, '\n%%... Best-Fit Solution\n');
fprintf(fid, '%% Source: opiFit_TwoWinds_OxygenClumped_BestFit.mat\n');
fprintf(fid, '%% Combined reduced chi-square: %.6f\n', S.chiR2Total);
fprintf(fid, '%% Combined degrees of freedom: %d\n', S.nuTotal);
fprintf(fid, '%% Clumped residual mean: %.3f C\n', S.detail.meanResidualT_C);
fprintf(fid, '%% Dolomite environment offset: %.1f +/- %.1f C\n', ...
    S.dolomiteOffsetC, S.sigmaOffsetC);
fprintf(fid, '%% Clumped comparison season: %s\n', S.clumpedSeason);
fprintf(fid, '%% Best-fit parameters:\n');
fprintf(fid, '%.8g\t', S.beta);
fprintf(fid, '\n');
end
