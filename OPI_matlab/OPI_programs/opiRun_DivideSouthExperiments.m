function opiRun_DivideSouthExperiments()
% opiRun_DivideSouthExperiments runs the Qiangtang divide-south suite.

scenarioRoot = ['/Users/keranli/Desktop/Coding/OPI_matlab/scenarios/', ...
    'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid'];
experimentRoot = fullfile(scenarioRoot, 'divide_south_experiments');
clumpedFile = fullfile(scenarioRoot, 'proxy_clumped', 'clumped_temperature.xlsx');

experiments = [
    struct('folder', 'south_010deg', ...
    'runFile', 'Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive_DivideSouth010.run')
    struct('folder', 'south_020deg', ...
    'runFile', 'Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive_DivideSouth020.run')
    struct('folder', 'south_030deg', ...
    'runFile', 'Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive_DivideSouth030.run')];

for i = 1:numel(experiments)
    runFile = fullfile(experimentRoot, experiments(i).folder, experiments(i).runFile);
    runPath = fileparts(runFile);
    fprintf('\n===== Running divide-south experiment %d/%d =====\n', i, numel(experiments));
    fprintf('%s\n', runFile);

    fitFile = fullfile(runPath, 'opiFit_TwoWinds_OxygenClumped_BestFit.mat');
    if exist(fitFile, 'file')
        fprintf('Reusing existing fit file:\n%s\n', fitFile);
    else
        opiFit_TwoWinds_OxygenClumped(runFile);
    end
    bestRunFile = writeBestRunFile(runFile, fitFile);

    matFile = fullfile(runPath, 'opiCalc_TwoWinds_OxygenOnly_Results.mat');
    if exist(matFile, 'file')
        fprintf('Reusing existing calc file:\n%s\n', matFile);
    else
        opiCalc_TwoWinds_OxygenOnly(bestRunFile);
    end
    opiCompare_ClumpedTemperature(matFile, clumpedFile, fullfile(runPath, 'proxy_clumped'));
    opiMaps_TwoWinds(matFile);
end

writeSummary(experimentRoot, experiments);
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
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
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

function writeSummary(experimentRoot, experiments)
n = numel(experiments);
folder = strings(n, 1);
shift = [-0.10; -0.20; -0.30];
chiR2Total = nan(n, 1);
chiR2O = nan(n, 1);
chi2T = nan(n, 1);
clumpedMeanResidual = nan(n, 1);
clumpedMeanZ = nan(n, 1);
T0_1_C = nan(n, 1);
T0_2_C = nan(n, 1);
U1 = nan(n, 1);
U2 = nan(n, 1);
M1 = nan(n, 1);
M2 = nan(n, 1);
fP1 = nan(n, 1);
fP2 = nan(n, 1);
fraction = nan(n, 1);
hrenWarmestResidualMean = nan(n, 1);
hrenWarmestZAbsMax = nan(n, 1);

for i = 1:n
    folder(i) = string(experiments(i).folder);
    runPath = fullfile(experimentRoot, experiments(i).folder);
    S = load(fullfile(runPath, 'opiFit_TwoWinds_OxygenClumped_BestFit.mat'));
    b = S.beta;
    chiR2Total(i) = S.chiR2Total;
    chiR2O(i) = S.detail.chiR2O;
    chi2T(i) = S.detail.chi2T;
    clumpedMeanResidual(i) = S.detail.meanResidualT_C;
    clumpedMeanZ(i) = S.detail.meanZT;
    T0_1_C(i) = b(3) - 273.15;
    T0_2_C(i) = b(13) - 273.15;
    U1(i) = b(1);
    U2(i) = b(11);
    M1(i) = b(4);
    M2(i) = b(14);
    fP1(i) = b(9);
    fP2(i) = b(19);
    fraction(i) = b(10);

    C = readtable(fullfile(runPath, 'proxy_clumped', ...
        'clumped_temperature_HrenSheldon2012_comparison.csv'));
    hrenWarmestResidualMean(i) = mean( ...
        C.residual_Tclumped_minus_OPI_Hren2012_Tw_warmest_C, 'omitnan');
    hrenWarmestZAbsMax(i) = max(abs( ...
        C.z_Tclumped_minus_OPI_Hren2012_Tw_warmest), [], 'omitnan');
end

summary = table(folder, shift, chiR2Total, chiR2O, chi2T, ...
    clumpedMeanResidual, clumpedMeanZ, T0_1_C, T0_2_C, U1, U2, ...
    M1, M2, fP1, fP2, fraction, hrenWarmestResidualMean, ...
    hrenWarmestZAbsMax);
writetable(summary, fullfile(experimentRoot, 'divide_south_experiment_summary.csv'));
disp(summary)
end
