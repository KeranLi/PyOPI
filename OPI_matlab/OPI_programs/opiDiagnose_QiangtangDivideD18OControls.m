function opiDiagnose_QiangtangDivideD18OControls(rootScenario)
% opiDiagnose_QiangtangDivideD18OControls diagnoses controls on central
% Qiangtang divide-area d18O using existing two-wind result files.
%
% The diagnostic is mechanism-oriented. It separates:
%   1) state-1 and state-2 precipitation d18O,
%   2) state precipitation mixing proportions,
%   3) combined precipitation-weighted d18O,
%   4) simple topographic and precipitation diagnostics
% inside sample-centered radius windows.

if nargin < 1 || strlength(string(rootScenario)) == 0
    rootScenario = fullfile(fileparts(mfilename('fullpath')), '..', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
end
rootScenario = char(string(rootScenario));
rootScenario = normalizeFolderPath(rootScenario);

outDir = fullfile(rootScenario, 'divide_d18O_controls');
if ~isfolder(outDir)
    mkdir(outDir);
end

resultFiles = collectResultFiles(rootScenario);
if isempty(resultFiles)
    error('No opiCalc_TwoWinds_OxygenOnly_Results.mat files found under %s', ...
        rootScenario);
end

radiusKm = [25; 50; 75; 100; 150; 200];
allRows = table();
for iFile = 1:numel(resultFiles)
    matFile = resultFiles(iFile);
    caseRows = diagnoseResultFile(matFile, rootScenario, radiusKm);
    allRows = [allRows; caseRows]; %#ok<AGROW>
end
allRows = addBaselineDeltas(allRows);

summaryFile = fullfile(outDir, 'central_divide_d18O_controls_by_radius.csv');
writetable(allRows, summaryFile);

rows50 = allRows(allRows.radius_km == 50, :);
writetable(rows50, fullfile(outDir, 'central_divide_d18O_controls_50km.csv'));

groupSummary = summarizeGroups(rows50);
writetable(groupSummary, fullfile(outDir, ...
    'central_divide_d18O_controls_50km_group_summary.csv'));

mechanismSummary = summarizeMechanisms(rows50);
writetable(mechanismSummary, fullfile(outDir, ...
    'central_divide_d18O_controls_50km_mechanism_summary.csv'));

writeReport(outDir, rows50, groupSummary, mechanismSummary);
plotControls(rows50, outDir);

fprintf('Wrote Qiangtang divide d18O-control diagnostics to:\n%s\n', outDir);
end

function resultFiles = collectResultFiles(rootScenario)
patterns = {
    fullfile(rootScenario, 'oxygen_clumped_ultra_aggressive', ...
    'opiCalc_TwoWinds_OxygenOnly_Results.mat')
    fullfile(rootScenario, 'sensitivity_parameter_local_clumped', '*', ...
    'opiCalc_TwoWinds_OxygenOnly_Results.mat')
    fullfile(rootScenario, 'sensitivity_mechanism_local_clumped', '*', ...
    'opiCalc_TwoWinds_OxygenOnly_Results.mat')
    fullfile(rootScenario, 'sensitivity_azimuth_fine_clumped', '*', ...
    'opiCalc_TwoWinds_OxygenOnly_Results.mat')
    fullfile(rootScenario, 'sensitivity_az2_transition_clumped', '*', ...
    'opiCalc_TwoWinds_OxygenOnly_Results.mat')
    fullfile(rootScenario, 'sensitivity_divide_calc_only_clumped', '*', ...
    'opiCalc_TwoWinds_OxygenOnly_Results.mat')
    fullfile(rootScenario, 'sensitivity_divide_shift_clumped', '*', ...
    'opiCalc_TwoWinds_OxygenOnly_Results.mat')
    fullfile(rootScenario, 'sensitivity_proxy_clumped', '*', ...
    'opiCalc_TwoWinds_OxygenOnly_Results.mat')
    };

resultFiles = strings(0, 1);
for i = 1:numel(patterns)
    files = dir(patterns{i});
    for j = 1:numel(files)
        resultFiles(end+1, 1) = string(fullfile(files(j).folder, files(j).name)); %#ok<AGROW>
    end
end
resultFiles = unique(resultFiles, 'stable');
end

function rows = diagnoseResultFile(matFile, rootScenario, radiusKm)
S = load(matFile, 'runPath', 'runTitle', 'dataPath', 'topoFile', ...
    'lon', 'lat', 'x', 'y', 'lon0', 'lat0', ...
    'sampleLon', 'sampleLat', 'sampleD18O', ...
    'pGrid', 'pGrid_1', 'pGrid_2', ...
    'd18OGrid', 'd18OGrid_1', 'd18OGrid_2', 'fractionPGrid', ...
    'beta', 'chiR2', 'nu', 'contDivideFile', 'isSampleSide01');
S.hGrid = loadTopographyForResult(S, matFile);

[groupName, caseName] = classifyCase(matFile, rootScenario);
mechanismGroup = classifyMechanism(groupName, caseName);
centerLon = mean(S.sampleLon, 'omitnan');
centerLat = mean(S.sampleLat, 'omitnan');
[centerX, centerY] = lonlat2xy(centerLon, centerLat, S.lon0, S.lat0);
obsMean = mean(S.sampleD18O, 'omitnan') * 1e3;

rows = table();
for i = 1:numel(radiusKm)
    R = radiusKm(i);
    metrics = radiusMetrics(S, centerX, centerY, R);
    beta = S.beta(:)';

    row = table(string(caseName), string(groupName), string(mechanismGroup), ...
        string(matFile), ...
        string(S.runPath), string(S.runTitle), R, centerLon, centerLat, ...
        obsMean, metrics.combinedD18O, metrics.state1D18O, ...
        metrics.state2D18O, metrics.mixedFromSeparate, ...
        metrics.combinedD18O - obsMean, ...
        metrics.state1D18O - metrics.state2D18O, ...
        metrics.state1PrecipFraction, metrics.totalP, metrics.p1Total, ...
        metrics.p2Total, metrics.pRatio12, metrics.meanFractionPGrid, ...
        metrics.meanElevation, metrics.pWeightedElevation, metrics.nWet, ...
        metrics.nAll, S.chiR2, S.nu, ...
        beta(1), beta(2), beta(3), beta(4), beta(7) * 1e3, ...
        beta(8) * 1e3, beta(9), beta(10), ...
        beta(11), beta(12), beta(13), beta(14), beta(17) * 1e3, ...
        beta(18) * 1e3, beta(19), ...
        string(getOptionalField(S, 'contDivideFile')), ...
        logicalMean(getOptionalField(S, 'isSampleSide01')), ...
        'VariableNames', {'case_name', 'group_name', 'mechanism_group', ...
        'mat_file', ...
        'run_path', 'run_title', 'radius_km', 'center_lon', ...
        'center_lat', 'observed_mean_d18O_permille', ...
        'combined_d18O_permille', 'state1_d18O_permille', ...
        'state2_d18O_permille', 'mixed_from_state_means_d18O_permille', ...
        'combined_minus_observed_permille', ...
        'state1_minus_state2_d18O_permille', ...
        'state1_precip_fraction', 'total_precip_weight', ...
        'state1_precip_weight', 'state2_precip_weight', ...
        'state1_to_state2_precip_ratio', 'mean_fractionPGrid', ...
        'mean_elevation_m', 'precip_weighted_elevation_m', ...
        'n_wet_cells', 'n_all_valid_cells', 'chiR2_oxygen', 'nu_oxygen', ...
        'U1', 'Az1_deg', 'T0_1_K', 'M1', 'd18O0_1_permille', ...
        'd18O_dLat1_permille_per_deg', 'fP1', 'fraction_parameter', ...
        'U2', 'Az2_deg', 'T0_2_K', 'M2', 'd18O0_2_permille', ...
        'd18O_dLat2_permille_per_deg', 'fP2', 'contDivideFile', ...
        'sample_side_state1_fraction'});
    rows = [rows; row]; %#ok<AGROW>
end
end

function hGrid = loadTopographyForResult(S, matFile)
if isfield(S, 'dataPath') && isfield(S, 'topoFile') ...
        && isfile(fullfile(S.dataPath, S.topoFile))
    topoPath = fullfile(S.dataPath, S.topoFile);
else
    topoPath = fullfile(fileparts(matFile), S.topoFile);
end

if ~isfile(topoPath)
    error('Topography file not found for result %s: %s', matFile, topoPath);
end
[~, ~, hGrid] = gridRead(topoPath);
hGrid(hGrid < 0) = 0;
end

function metrics = radiusMetrics(S, centerX, centerY, radiusKm)
radiusM = radiusKm * 1e3;
[X, Y] = meshgrid(S.x, S.y);
inRadius = (X - centerX).^2 + (Y - centerY).^2 <= radiusM^2;

validCombined = inRadius & isfinite(S.d18OGrid) & isfinite(S.pGrid) & S.pGrid > 0;
valid1 = inRadius & isfinite(S.d18OGrid_1) & isfinite(S.pGrid_1) & S.pGrid_1 > 0;
valid2 = inRadius & isfinite(S.d18OGrid_2) & isfinite(S.pGrid_2) & S.pGrid_2 > 0;
validTopo = inRadius & isfinite(S.hGrid);

metrics.combinedD18O = weightedMeanPermil(S.d18OGrid, S.pGrid, validCombined);
metrics.state1D18O = weightedMeanPermil(S.d18OGrid_1, S.pGrid_1, valid1);
metrics.state2D18O = weightedMeanPermil(S.d18OGrid_2, S.pGrid_2, valid2);

metrics.p1Total = sum(S.pGrid_1(valid1), 'all');
metrics.p2Total = sum(S.pGrid_2(valid2), 'all');
metrics.totalP = sum(S.pGrid(validCombined), 'all');
if metrics.p1Total + metrics.p2Total > 0
    metrics.state1PrecipFraction = metrics.p1Total ...
        / (metrics.p1Total + metrics.p2Total);
else
    metrics.state1PrecipFraction = nan;
end
if metrics.p2Total > 0
    metrics.pRatio12 = metrics.p1Total / metrics.p2Total;
else
    metrics.pRatio12 = nan;
end

metrics.mixedFromSeparate = metrics.state1PrecipFraction * metrics.state1D18O ...
    + (1 - metrics.state1PrecipFraction) * metrics.state2D18O;

validFraction = inRadius & isfinite(S.fractionPGrid);
metrics.meanFractionPGrid = mean(S.fractionPGrid(validFraction), 'omitnan');
metrics.meanElevation = mean(S.hGrid(validTopo), 'omitnan');
metrics.pWeightedElevation = weightedMeanRaw(S.hGrid, S.pGrid, ...
    validCombined & isfinite(S.hGrid));
metrics.nWet = sum(validCombined, 'all');
metrics.nAll = sum(inRadius & isfinite(S.d18OGrid), 'all');
end

function value = weightedMeanPermil(valueGrid, weightGrid, valid)
if any(valid, 'all')
    totalWeight = sum(weightGrid(valid), 'all');
    if totalWeight > 0
        value = sum(weightGrid(valid) .* valueGrid(valid), 'all') ...
            / totalWeight * 1e3;
    else
        value = nan;
    end
else
    value = nan;
end
end

function value = weightedMeanRaw(valueGrid, weightGrid, valid)
if any(valid, 'all')
    totalWeight = sum(weightGrid(valid), 'all');
    if totalWeight > 0
        value = sum(weightGrid(valid) .* valueGrid(valid), 'all') / totalWeight;
    else
        value = nan;
    end
else
    value = nan;
end
end

function [groupName, caseName] = classifyCase(matFile, rootScenario)
matFile = string(normalizeFilePath(matFile));
rootScenario = string(normalizeFolderPath(rootScenario));
rel = erase(matFile, rootScenario + filesep);
parts = split(rel, filesep);

if startsWith(rel, "oxygen_clumped_ultra_aggressive")
    groupName = "baseline";
    caseName = "baseline";
elseif numel(parts) >= 2 && startsWith(parts(1), "sensitivity_")
    groupFolder = parts(1);
    caseName = parts(2);
    if contains(groupFolder, "parameter")
        groupName = "parameter";
    elseif contains(groupFolder, "mechanism")
        groupName = "mechanism";
    elseif contains(groupFolder, "azimuth_fine")
        groupName = "azimuth_fine";
    elseif contains(groupFolder, "az2_transition")
        groupName = "az2_transition";
    elseif contains(groupFolder, "divide_calc_only")
        groupName = "divide_calc_only";
    elseif contains(groupFolder, "divide")
        groupName = "divide";
    elseif contains(groupFolder, "proxy")
        groupName = "proxy";
    else
        groupName = groupFolder;
    end
else
    groupName = "other";
    caseName = parts(1);
end
end

function mechanismGroup = classifyMechanism(groupName, caseName)
caseName = string(caseName);
groupName = string(groupName);

if groupName == "baseline"
    mechanismGroup = "baseline";
elseif groupName == "parameter" && startsWith(caseName, "M_1_")
    mechanismGroup = "state1_rainout_M";
elseif groupName == "parameter" && startsWith(caseName, "M_2_")
    mechanismGroup = "state2_rainout_M";
elseif groupName == "parameter" && startsWith(caseName, "T0_1_")
    mechanismGroup = "state1_temperature_T0";
elseif groupName == "parameter" && startsWith(caseName, "T0_2_")
    mechanismGroup = "state2_temperature_T0";
elseif groupName == "parameter" && startsWith(caseName, "fraction_")
    mechanismGroup = "wind_state_mixing_fraction";
elseif groupName == "mechanism" && startsWith(caseName, "d18O0_1_")
    mechanismGroup = "state1_source_d18O0";
elseif groupName == "mechanism" && startsWith(caseName, "d18O0_2_")
    mechanismGroup = "state2_source_d18O0";
elseif groupName == "mechanism" && startsWith(caseName, "d18Olat1_")
    mechanismGroup = "state1_source_latitude_gradient";
elseif groupName == "mechanism" && startsWith(caseName, "d18Olat2_")
    mechanismGroup = "state2_source_latitude_gradient";
elseif groupName == "mechanism" && startsWith(caseName, "fP1_")
    mechanismGroup = "state1_evaporation_fP";
elseif groupName == "mechanism" && startsWith(caseName, "fP2_")
    mechanismGroup = "state2_evaporation_fP";
elseif groupName == "mechanism" && startsWith(caseName, "Az1_")
    mechanismGroup = "state1_wind_path_azimuth";
elseif groupName == "mechanism" && startsWith(caseName, "Az2_")
    mechanismGroup = "state2_wind_path_azimuth";
elseif groupName == "azimuth_fine" && startsWith(caseName, "Az1_")
    mechanismGroup = "azimuth_fine_state1";
elseif groupName == "azimuth_fine" && startsWith(caseName, "Az2_")
    mechanismGroup = "azimuth_fine_state2";
elseif groupName == "az2_transition"
    mechanismGroup = "az2_transition";
elseif groupName == "divide_calc_only"
    mechanismGroup = "divide_position_calc_only";
elseif groupName == "divide"
    mechanismGroup = "divide_position_refit";
elseif groupName == "proxy"
    mechanismGroup = "proxy_choice_refit";
else
    mechanismGroup = groupName;
end
end

function rows = addBaselineDeltas(rows)
newNames = {'delta_combined_d18O_vs_baseline_permille', ...
    'delta_state1_d18O_vs_baseline_permille', ...
    'delta_state2_d18O_vs_baseline_permille', ...
    'delta_state1_precip_fraction_vs_baseline', ...
    'mixing_fraction_contribution_permille', ...
    'state1_isotope_contribution_permille', ...
    'state2_isotope_contribution_permille', ...
    'decomposition_residual_permille'};
for iName = 1:numel(newNames)
    rows.(newNames{iName}) = nan(height(rows), 1);
end

radii = unique(rows.radius_km, 'stable');
for iRadius = 1:numel(radii)
    r = radii(iRadius);
    baseIdx = rows.group_name == "baseline" & rows.radius_km == r;
    if ~any(baseIdx)
        continue
    end
    base = rows(find(baseIdx, 1, 'first'), :);
    idx = rows.radius_km == r;

    f0 = base.state1_precip_fraction;
    d10 = base.state1_d18O_permille;
    d20 = base.state2_d18O_permille;
    combined0 = base.combined_d18O_permille;

    f = rows.state1_precip_fraction(idx);
    d1 = rows.state1_d18O_permille(idx);
    d2 = rows.state2_d18O_permille(idx);
    totalDelta = rows.combined_d18O_permille(idx) - combined0;

    % Exact midpoint decomposition of:
    % d18O = f * d18O_1 + (1 - f) * d18O_2.
    % This avoids treating the interaction between changing f and changing
    % state isotope values as an unexplained residual.
    fMid = 0.5 .* (f + f0);
    d1Mid = 0.5 .* (d1 + d10);
    d2Mid = 0.5 .* (d2 + d20);
    mixingContribution = (f - f0) .* (d1Mid - d2Mid);
    state1Contribution = fMid .* (d1 - d10);
    state2Contribution = (1 - fMid) .* (d2 - d20);

    rows.delta_combined_d18O_vs_baseline_permille(idx) = totalDelta;
    rows.delta_state1_d18O_vs_baseline_permille(idx) = d1 - d10;
    rows.delta_state2_d18O_vs_baseline_permille(idx) = d2 - d20;
    rows.delta_state1_precip_fraction_vs_baseline(idx) = f - f0;
    rows.mixing_fraction_contribution_permille(idx) = mixingContribution;
    rows.state1_isotope_contribution_permille(idx) = state1Contribution;
    rows.state2_isotope_contribution_permille(idx) = state2Contribution;
    rows.decomposition_residual_permille(idx) = totalDelta ...
        - mixingContribution - state1Contribution - state2Contribution;
end
end

function normalizedPath = normalizeFolderPath(pathIn)
oldDir = pwd;
cleanup = onCleanup(@() cd(oldDir)); %#ok<NASGU>
cd(char(pathIn));
normalizedPath = pwd;
end

function normalizedPath = normalizeFilePath(pathIn)
[folderPath, fileName, fileExt] = fileparts(char(pathIn));
folderPath = normalizeFolderPath(folderPath);
normalizedPath = fullfile(folderPath, [fileName, fileExt]);
end

function value = getOptionalField(S, fieldName)
if isfield(S, fieldName)
    value = S.(fieldName);
else
    value = [];
end
end

function value = logicalMean(x)
if isempty(x)
    value = nan;
else
    value = mean(logical(x), 'omitnan');
end
end

function groupSummary = summarizeGroups(rows50)
groups = unique(rows50.group_name, 'stable');
groupSummary = table();
for i = 1:numel(groups)
    G = rows50(rows50.group_name == groups(i), :);
    row = table(groups(i), height(G), ...
        min(G.combined_d18O_permille, [], 'omitnan'), ...
        max(G.combined_d18O_permille, [], 'omitnan'), ...
        rangeOmitnan(G.combined_d18O_permille), ...
        rangeOmitnan(G.state1_d18O_permille), ...
        rangeOmitnan(G.state2_d18O_permille), ...
        rangeOmitnan(G.state1_precip_fraction), ...
        rangeOmitnan(G.state1_minus_state2_d18O_permille), ...
        rangeOmitnan(G.precip_weighted_elevation_m), ...
        'VariableNames', {'group_name', 'n_cases', ...
        'combined_d18O_min_permille', 'combined_d18O_max_permille', ...
        'combined_d18O_span_permille', 'state1_d18O_span_permille', ...
        'state2_d18O_span_permille', ...
        'state1_precip_fraction_span', ...
        'state1_minus_state2_span_permille', ...
        'precip_weighted_elevation_span_m'});
    groupSummary = [groupSummary; row]; %#ok<AGROW>
end
end

function mechanismSummary = summarizeMechanisms(rows50)
groups = unique(rows50.mechanism_group, 'stable');
mechanismSummary = table();
for i = 1:numel(groups)
    G = rows50(rows50.mechanism_group == groups(i), :);
    row = table(groups(i), height(G), ...
        maxAbsOmitnan(G.delta_combined_d18O_vs_baseline_permille), ...
        maxAbsOmitnan(G.mixing_fraction_contribution_permille), ...
        maxAbsOmitnan(G.state1_isotope_contribution_permille), ...
        maxAbsOmitnan(G.state2_isotope_contribution_permille), ...
        maxAbsOmitnan(G.decomposition_residual_permille), ...
        rangeOmitnan(G.combined_d18O_permille), ...
        rangeOmitnan(G.state1_precip_fraction), ...
        'VariableNames', {'mechanism_group', 'n_cases', ...
        'max_abs_total_delta_permille', ...
        'max_abs_mixing_fraction_contribution_permille', ...
        'max_abs_state1_isotope_contribution_permille', ...
        'max_abs_state2_isotope_contribution_permille', ...
        'max_abs_decomposition_residual_permille', ...
        'combined_d18O_span_permille', ...
        'state1_precip_fraction_span'});
    mechanismSummary = [mechanismSummary; row]; %#ok<AGROW>
end
end

function value = rangeOmitnan(x)
if isempty(x) || all(~isfinite(x))
    value = nan;
else
    value = max(x, [], 'omitnan') - min(x, [], 'omitnan');
end
end

function value = maxAbsOmitnan(x)
if isempty(x) || all(~isfinite(x))
    value = nan;
else
    value = max(abs(x), [], 'omitnan');
end
end

function writeReport(outDir, rows50, groupSummary, mechanismSummary)
fid = fopen(fullfile(outDir, 'central_divide_d18O_controls_report.md'), ...
    'w', 'native', 'UTF-8');
if fid == -1
    error('Could not write mechanism report.');
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, '# Central Qiangtang divide d18O-control diagnostics\n\n');
fprintf(fid, 'Generated by `opiDiagnose_QiangtangDivideD18OControls`.\n\n');
fprintf(fid, 'The 50 km sample-centered window is used for the main summary.\n\n');
fprintf(fid, ['Contribution columns use an exact midpoint decomposition of ' ...
    '`d18O = f * d18O_1 + (1 - f) * d18O_2`, relative to the baseline ' ...
    'at the same radius.\n\n']);

base = rows50(rows50.group_name == "baseline", :);
if ~isempty(base)
    fprintf(fid, '## Baseline 50 km decomposition\n\n');
    fprintf(fid, '- Combined d18O: %.3f per mil\n', base.combined_d18O_permille(1));
    fprintf(fid, '- State 1 d18O: %.3f per mil\n', base.state1_d18O_permille(1));
    fprintf(fid, '- State 2 d18O: %.3f per mil\n', base.state2_d18O_permille(1));
    fprintf(fid, '- State 1 precipitation fraction: %.3f\n', ...
        base.state1_precip_fraction(1));
    fprintf(fid, '- State 1 minus state 2 d18O: %.3f per mil\n\n', ...
        base.state1_minus_state2_d18O_permille(1));
end

fprintf(fid, '## Mechanism summary at 50 km\n\n');
for i = 1:height(mechanismSummary)
    R = mechanismSummary(i, :);
    fprintf(fid, ['- `%s`: max |total delta| %.3f per mil; ' ...
        'max |mixing contribution| %.3f; max |state1 isotope contribution| %.3f; ' ...
        'max |state2 isotope contribution| %.3f.\n'], ...
        R.mechanism_group, R.max_abs_total_delta_permille, ...
        R.max_abs_mixing_fraction_contribution_permille, ...
        R.max_abs_state1_isotope_contribution_permille, ...
        R.max_abs_state2_isotope_contribution_permille);
end

fprintf(fid, '\n## Largest 50 km shifts relative to baseline\n\n');
ranked = rows50(rows50.group_name ~= "baseline", :);
if ~isempty(ranked)
    [~, order] = sort(abs(ranked.delta_combined_d18O_vs_baseline_permille), ...
        'descend');
    ranked = ranked(order, :);
    nTop = min(8, height(ranked));
    for i = 1:nTop
        R = ranked(i, :);
        fprintf(fid, ['- `%s` (`%s`): total delta %.3f per mil; ' ...
            'mixing %.3f; state1 isotope %.3f; state2 isotope %.3f; ' ...
            'state1 precip fraction %.3f.\n'], ...
            R.case_name, R.mechanism_group, ...
            R.delta_combined_d18O_vs_baseline_permille, ...
            R.mixing_fraction_contribution_permille, ...
            R.state1_isotope_contribution_permille, ...
            R.state2_isotope_contribution_permille, ...
            R.state1_precip_fraction);
    end
end

fprintf(fid, '\n## Working interpretation\n\n');
fprintf(fid, ['- The baseline central divide signal is dominated by state 1 ' ...
    'precipitation: state 1 supplies %.1f%% of 50 km precipitation, with ' ...
    'state 1 at %.3f per mil and state 2 at %.3f per mil.\n'], ...
    100 * base.state1_precip_fraction(1), ...
    base.state1_d18O_permille(1), base.state2_d18O_permille(1));
fprintf(fid, ['- Therefore the strongest direct control in the current ' ...
    'calc-only parameter suite is the wind-state mixing fraction. It changes ' ...
    'central divide d18O almost entirely by changing how much depleted ' ...
    'state 2 precipitation enters the local precipitation-weighted mean.\n']);
fprintf(fid, ['- State 1 temperature and rainout mainly act by changing state 1 ' ...
    'd18O. State 2 changes can look large in state-2-only d18O, but their ' ...
    'effect on combined d18O is damped when state 2 contributes little ' ...
    'precipitation.\n']);
fprintf(fid, ['- Divide-position and proxy-choice cases are refit experiments. ' ...
    'Their shifts should be read as parameter-compensation outcomes under ' ...
    'changed assumptions, not as pure divide-geometry or pure proxy-season ' ...
    'forcing.\n']);

fprintf(fid, '## Group spans at 50 km\n\n');
for i = 1:height(groupSummary)
    R = groupSummary(i, :);
    fprintf(fid, ['- `%s`: combined d18O span %.3f per mil; ' ...
        'state1 d18O span %.3f; state2 d18O span %.3f; ' ...
        'state1 precip-fraction span %.3f.\n'], ...
        R.group_name, R.combined_d18O_span_permille, ...
        R.state1_d18O_span_permille, R.state2_d18O_span_permille, ...
        R.state1_precip_fraction_span);
end

fprintf(fid, '\n## Output files\n\n');
fprintf(fid, '- `central_divide_d18O_controls_by_radius.csv`\n');
fprintf(fid, '- `central_divide_d18O_controls_50km.csv`\n');
fprintf(fid, '- `central_divide_d18O_controls_50km_group_summary.csv`\n');
fprintf(fid, '- `central_divide_d18O_controls_50km_mechanism_summary.csv`\n');
fprintf(fid, '- `central_divide_d18O_controls_50km_scatter.png`\n');
end

function plotControls(rows50, outDir)
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 480]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile
scatterByGroup(rows50.state1_precip_fraction, rows50.combined_d18O_permille, ...
    rows50.group_name);
grid on
xlabel('State 1 precipitation fraction, 50 km');
ylabel('Combined d18O, 50 km (per mil)');
title('Mixing control');

nexttile
scatterByGroup(rows50.state1_minus_state2_d18O_permille, ...
    rows50.combined_d18O_permille, rows50.group_name);
grid on
xlabel('State 1 - state 2 d18O (per mil)');
ylabel('Combined d18O, 50 km (per mil)');
title('Source isotope contrast');

exportgraphics(fig, fullfile(outDir, ...
    'central_divide_d18O_controls_50km_scatter.png'), 'Resolution', 220);
close(fig);
end

function scatterByGroup(x, y, groupName)
groups = unique(groupName, 'stable');
colors = lines(numel(groups));
hold on
for i = 1:numel(groups)
    idx = groupName == groups(i) & isfinite(x) & isfinite(y);
    scatter(x(idx), y(idx), 42, colors(i, :), 'filled', ...
        'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
end
legend(cellstr(groups), 'Location', 'best', 'Interpreter', 'none');
hold off
end
