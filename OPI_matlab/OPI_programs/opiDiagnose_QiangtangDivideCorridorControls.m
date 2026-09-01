function opiDiagnose_QiangtangDivideCorridorControls(rootScenario)
% opiDiagnose_QiangtangDivideCorridorControls diagnoses d18O controls along
% the central Qiangtang divide corridor.
%
% This diagnostic complements the sample-centered circular-window analysis.
% It uses the continental-divide polyline from each result file and computes
% precipitation-weighted d18O controls inside divide-centered corridor
% buffers.

if nargin < 1 || strlength(string(rootScenario)) == 0
    rootScenario = fullfile(fileparts(mfilename('fullpath')), '..', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
end
rootScenario = char(string(rootScenario));
rootScenario = normalizeFolderPath(rootScenario);

outDir = fullfile(rootScenario, 'divide_corridor_d18O_controls');
if ~isfolder(outDir)
    mkdir(outDir);
end

resultFiles = collectResultFiles(rootScenario);
if isempty(resultFiles)
    error('No opiCalc_TwoWinds_OxygenOnly_Results.mat files found under %s', ...
        rootScenario);
end

corridorHalfWidthKm = [25; 50; 75; 100];
allRows = table();
for iFile = 1:numel(resultFiles)
    matFile = resultFiles(iFile);
    caseRows = diagnoseResultFile(matFile, rootScenario, corridorHalfWidthKm);
    allRows = [allRows; caseRows]; %#ok<AGROW>
end
allRows = addBaselineDeltas(allRows);

writetable(allRows, fullfile(outDir, ...
    'central_divide_corridor_d18O_controls_by_width.csv'));

rows50 = allRows(allRows.corridor_half_width_km == 50, :);
writetable(rows50, fullfile(outDir, ...
    'central_divide_corridor_d18O_controls_50km.csv'));

groupSummary = summarizeGroups(rows50);
writetable(groupSummary, fullfile(outDir, ...
    'central_divide_corridor_d18O_controls_50km_group_summary.csv'));

mechanismSummary = summarizeMechanisms(rows50);
writetable(mechanismSummary, fullfile(outDir, ...
    'central_divide_corridor_d18O_controls_50km_mechanism_summary.csv'));

writeReport(outDir, rows50, groupSummary, mechanismSummary);
plotCorridorControls(rows50, outDir);

fprintf('Wrote Qiangtang divide-corridor d18O-control diagnostics to:\n%s\n', outDir);
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

function rows = diagnoseResultFile(matFile, rootScenario, corridorHalfWidthKm)
S = load(matFile, 'runPath', 'runTitle', 'dataPath', 'topoFile', ...
    'lon', 'lat', 'x', 'y', 'lon0', 'lat0', ...
    'sampleD18O', 'pGrid', 'pGrid_1', 'pGrid_2', ...
    'd18OGrid', 'd18OGrid_1', 'd18OGrid_2', 'fractionPGrid', ...
    'beta', 'chiR2', 'nu', 'contDivideFile');
S.hGrid = loadTopographyForResult(S, matFile);

[groupName, caseName] = classifyCase(matFile, rootScenario);
mechanismGroup = classifyMechanism(groupName, caseName);
[divideX, divideY] = loadDivideXY(S, matFile);
[X, Y] = meshgrid(S.x, S.y);
distanceToDivideM = distanceToPolyline(X, Y, divideX, divideY);
obsMean = mean(S.sampleD18O, 'omitnan') * 1e3;

rows = table();
for i = 1:numel(corridorHalfWidthKm)
    W = corridorHalfWidthKm(i);
    metrics = corridorMetrics(S, distanceToDivideM, W);
    beta = S.beta(:)';

    row = table(string(caseName), string(groupName), string(mechanismGroup), ...
        string(matFile), string(S.runPath), string(S.runTitle), W, ...
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
        'VariableNames', {'case_name', 'group_name', 'mechanism_group', ...
        'mat_file', 'run_path', 'run_title', 'corridor_half_width_km', ...
        'observed_mean_d18O_permille', 'combined_d18O_permille', ...
        'state1_d18O_permille', 'state2_d18O_permille', ...
        'mixed_from_state_means_d18O_permille', ...
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
        'd18O_dLat2_permille_per_deg', 'fP2', 'contDivideFile'});
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

function [divideX, divideY] = loadDivideXY(S, matFile)
if isfield(S, 'dataPath') && isfield(S, 'contDivideFile') ...
        && isfile(fullfile(S.dataPath, S.contDivideFile))
    dividePath = fullfile(S.dataPath, S.contDivideFile);
else
    dividePath = fullfile(fileparts(matFile), S.contDivideFile);
end
if ~isfile(dividePath)
    error('Divide file not found for result %s: %s', matFile, dividePath);
end
D = load(dividePath, 'contDivideLon', 'contDivideLat');
[divideX, divideY] = lonlat2xy(D.contDivideLon, D.contDivideLat, ...
    S.lon0, S.lat0);
divideX = divideX(:);
divideY = divideY(:);
end

function distanceM = distanceToPolyline(X, Y, lineX, lineY)
distanceM = inf(size(X));
for i = 1:(numel(lineX) - 1)
    x1 = lineX(i);
    y1 = lineY(i);
    x2 = lineX(i + 1);
    y2 = lineY(i + 1);
    dx = x2 - x1;
    dy = y2 - y1;
    denom = dx^2 + dy^2;
    if denom == 0
        d = hypot(X - x1, Y - y1);
    else
        t = ((X - x1) .* dx + (Y - y1) .* dy) ./ denom;
        t = max(0, min(1, t));
        projX = x1 + t .* dx;
        projY = y1 + t .* dy;
        d = hypot(X - projX, Y - projY);
    end
    distanceM = min(distanceM, d);
end
end

function metrics = corridorMetrics(S, distanceToDivideM, corridorHalfWidthKm)
inCorridor = distanceToDivideM <= corridorHalfWidthKm * 1e3;

validCombined = inCorridor & isfinite(S.d18OGrid) & isfinite(S.pGrid) & S.pGrid > 0;
valid1 = inCorridor & isfinite(S.d18OGrid_1) & isfinite(S.pGrid_1) & S.pGrid_1 > 0;
valid2 = inCorridor & isfinite(S.d18OGrid_2) & isfinite(S.pGrid_2) & S.pGrid_2 > 0;
validTopo = inCorridor & isfinite(S.hGrid);

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
validFraction = inCorridor & isfinite(S.fractionPGrid);
metrics.meanFractionPGrid = mean(S.fractionPGrid(validFraction), 'omitnan');
metrics.meanElevation = mean(S.hGrid(validTopo), 'omitnan');
metrics.pWeightedElevation = weightedMeanRaw(S.hGrid, S.pGrid, ...
    validCombined & isfinite(S.hGrid));
metrics.nWet = sum(validCombined, 'all');
metrics.nAll = sum(inCorridor & isfinite(S.d18OGrid), 'all');
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

widths = unique(rows.corridor_half_width_km, 'stable');
for iWidth = 1:numel(widths)
    w = widths(iWidth);
    baseIdx = rows.group_name == "baseline" & rows.corridor_half_width_km == w;
    if ~any(baseIdx)
        continue
    end
    base = rows(find(baseIdx, 1, 'first'), :);
    idx = rows.corridor_half_width_km == w;

    f0 = base.state1_precip_fraction;
    d10 = base.state1_d18O_permille;
    d20 = base.state2_d18O_permille;
    combined0 = base.combined_d18O_permille;

    f = rows.state1_precip_fraction(idx);
    d1 = rows.state1_d18O_permille(idx);
    d2 = rows.state2_d18O_permille(idx);
    totalDelta = rows.combined_d18O_permille(idx) - combined0;

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

function groupSummary = summarizeGroups(rows50)
groups = unique(rows50.group_name, 'stable');
groupSummary = table();
for i = 1:numel(groups)
    G = rows50(rows50.group_name == groups(i), :);
    row = table(groups(i), height(G), ...
        rangeOmitnan(G.combined_d18O_permille), ...
        rangeOmitnan(G.state1_d18O_permille), ...
        rangeOmitnan(G.state2_d18O_permille), ...
        rangeOmitnan(G.state1_precip_fraction), ...
        maxAbsOmitnan(G.delta_combined_d18O_vs_baseline_permille), ...
        'VariableNames', {'group_name', 'n_cases', ...
        'combined_d18O_span_permille', 'state1_d18O_span_permille', ...
        'state2_d18O_span_permille', ...
        'state1_precip_fraction_span', ...
        'max_abs_total_delta_permille'});
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
        rangeOmitnan(G.combined_d18O_permille), ...
        rangeOmitnan(G.state1_precip_fraction), ...
        'VariableNames', {'mechanism_group', 'n_cases', ...
        'max_abs_total_delta_permille', ...
        'max_abs_mixing_fraction_contribution_permille', ...
        'max_abs_state1_isotope_contribution_permille', ...
        'max_abs_state2_isotope_contribution_permille', ...
        'combined_d18O_span_permille', ...
        'state1_precip_fraction_span'});
    mechanismSummary = [mechanismSummary; row]; %#ok<AGROW>
end
end

function writeReport(outDir, rows50, groupSummary, mechanismSummary)
fid = fopen(fullfile(outDir, 'central_divide_corridor_d18O_controls_report.md'), ...
    'w', 'native', 'UTF-8');
if fid == -1
    error('Could not write corridor mechanism report.');
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, '# Central Qiangtang divide-corridor d18O controls\n\n');
fprintf(fid, 'Generated by `opiDiagnose_QiangtangDivideCorridorControls`.\n\n');
fprintf(fid, 'The main summary uses a +/-50 km corridor around each case divide line.\n\n');

base = rows50(rows50.group_name == "baseline", :);
if ~isempty(base)
    fprintf(fid, '## Baseline +/-50 km corridor decomposition\n\n');
    fprintf(fid, '- Combined d18O: %.3f per mil\n', base.combined_d18O_permille(1));
    fprintf(fid, '- State 1 d18O: %.3f per mil\n', base.state1_d18O_permille(1));
    fprintf(fid, '- State 2 d18O: %.3f per mil\n', base.state2_d18O_permille(1));
    fprintf(fid, '- State 1 precipitation fraction: %.3f\n\n', ...
        base.state1_precip_fraction(1));
end

fprintf(fid, '## Mechanism summary, +/-50 km corridor\n\n');
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

fprintf(fid, '\n## Largest +/-50 km corridor shifts relative to baseline\n\n');
ranked = rows50(rows50.group_name ~= "baseline", :);
if ~isempty(ranked)
    [~, order] = sort(abs(ranked.delta_combined_d18O_vs_baseline_permille), ...
        'descend');
    ranked = ranked(order, :);
    nTop = min(10, height(ranked));
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

fprintf(fid, '\n## Group spans, +/-50 km corridor\n\n');
for i = 1:height(groupSummary)
    R = groupSummary(i, :);
    fprintf(fid, ['- `%s`: combined d18O span %.3f per mil; ' ...
        'state1 precip-fraction span %.3f.\n'], ...
        R.group_name, R.combined_d18O_span_permille, ...
        R.state1_precip_fraction_span);
end

fprintf(fid, '\n## Working interpretation\n\n');
fprintf(fid, ['- The +/-50 km corridor baseline is more state-1 dominated ' ...
    'than the sample-centered 50 km circular window: state 1 supplies ' ...
    '%.1f%% of corridor precipitation.\n'], ...
    100 * base.state1_precip_fraction(1));
fprintf(fid, ['- Because the corridor is strongly state-1 dominated, ' ...
    'state-1 source d18O0 and state-1 isotope-path changes are stronger ' ...
    'controls in the corridor summary than in the circular-window summary.\n']);
fprintf(fid, ['- Divide calc-only cases can differ in the corridor diagnostic ' ...
    'because the corridor is rebuilt around each shifted divide line. This ' ...
    'tests a moving spatial corridor, not a change in the isotope grid at ' ...
    'fixed coordinates.\n']);

fprintf(fid, '\n## Output files\n\n');
fprintf(fid, '- `central_divide_corridor_d18O_controls_by_width.csv`\n');
fprintf(fid, '- `central_divide_corridor_d18O_controls_50km.csv`\n');
fprintf(fid, '- `central_divide_corridor_d18O_controls_50km_group_summary.csv`\n');
fprintf(fid, '- `central_divide_corridor_d18O_controls_50km_mechanism_summary.csv`\n');
fprintf(fid, '- `central_divide_corridor_d18O_controls_50km_scatter.png`\n');
end

function plotCorridorControls(rows50, outDir)
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 480]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile
scatterByGroup(rows50.state1_precip_fraction, rows50.combined_d18O_permille, ...
    rows50.group_name);
grid on
xlabel('State 1 precipitation fraction, +/-50 km corridor');
ylabel('Combined d18O (per mil)');
title('Corridor mixing control');

nexttile
scatterByGroup(rows50.delta_state1_precip_fraction_vs_baseline, ...
    rows50.delta_combined_d18O_vs_baseline_permille, rows50.group_name);
grid on
xlabel('Delta state 1 precipitation fraction');
ylabel('Delta combined d18O (per mil)');
title('Corridor delta response');

exportgraphics(fig, fullfile(outDir, ...
    'central_divide_corridor_d18O_controls_50km_scatter.png'), ...
    'Resolution', 220);
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

function value = getOptionalField(S, fieldName)
if isfield(S, fieldName)
    value = S.(fieldName);
else
    value = [];
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
