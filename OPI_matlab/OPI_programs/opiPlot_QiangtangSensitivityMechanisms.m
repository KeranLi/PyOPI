function opiPlot_QiangtangSensitivityMechanisms(rootScenario)
% opiPlot_QiangtangSensitivityMechanisms plots central-divide d18O controls.
%
% The script uses outputs from opiDiagnose_QiangtangDivideD18OControls and
% creates publication-oriented summary figures for the 50 km diagnostic
% window.

if nargin < 1 || strlength(string(rootScenario)) == 0
    rootScenario = fullfile(fileparts(mfilename('fullpath')), '..', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
end
rootScenario = char(string(rootScenario));
controlsDir = fullfile(rootScenario, 'divide_d18O_controls');

rowsFile = fullfile(controlsDir, 'central_divide_d18O_controls_50km.csv');
summaryFile = fullfile(controlsDir, ...
    'central_divide_d18O_controls_50km_mechanism_summary.csv');
if ~isfile(rowsFile)
    error('Missing 50 km controls table: %s', rowsFile);
end
if ~isfile(summaryFile)
    error('Missing mechanism summary table: %s', summaryFile);
end

rows50 = readtable(rowsFile, 'TextType', 'string');
mechanismSummary = readtable(summaryFile, 'TextType', 'string');

plotMechanismRank(mechanismSummary, controlsDir);
plotTopCaseContributions(rows50, controlsDir);
azTable = buildAzimuthResponseTable(rows50);
writetable(azTable, fullfile(controlsDir, ...
    'central_divide_azimuth_fine_response_50km.csv'));
plotAzimuthResponse(azTable, controlsDir);
az2Table = buildAz2TransitionResponseTable(rows50);
writetable(az2Table, fullfile(controlsDir, ...
    'central_divide_az2_transition_response_50km.csv'));
plotAz2TransitionResponse(az2Table, controlsDir);

fprintf('Wrote Qiangtang mechanism plots to:\n%s\n', controlsDir);
end

function plotMechanismRank(mechanismSummary, controlsDir)
T = mechanismSummary(mechanismSummary.mechanism_group ~= "baseline", :);
T = T(isfinite(T.max_abs_total_delta_permille), :);
[~, order] = sort(T.max_abs_total_delta_permille, 'ascend');
T = T(order, :);

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 960 720]);
barh(T.max_abs_total_delta_permille, 'FaceColor', [0.20 0.45 0.70], ...
    'EdgeColor', 'none');
set(gca, 'YTick', 1:height(T), ...
    'YTickLabel', cellstr(strrep(T.mechanism_group, "_", " ")), ...
    'TickLabelInterpreter', 'none');
grid on
xlabel('Max absolute total delta d18O at 50 km (per mil)');
title('Central Qiangtang divide mechanism ranking');
exportgraphics(fig, fullfile(controlsDir, ...
    'central_divide_mechanism_ranked_delta_50km.png'), 'Resolution', 240);
close(fig);
end

function plotTopCaseContributions(rows50, controlsDir)
T = rows50(rows50.group_name ~= "baseline", :);
T = T(isfinite(T.delta_combined_d18O_vs_baseline_permille), :);
[~, order] = sort(abs(T.delta_combined_d18O_vs_baseline_permille), 'descend');
nTop = min(12, height(T));
T = T(order(1:nTop), :);
T = flipud(T);

Y = [T.mixing_fraction_contribution_permille, ...
    T.state1_isotope_contribution_permille, ...
    T.state2_isotope_contribution_permille];

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 760]);
barh(Y, 'stacked', 'EdgeColor', 'none');
hold on
plot(T.delta_combined_d18O_vs_baseline_permille, 1:height(T), ...
    'ko', 'MarkerFaceColor', 'w', 'LineWidth', 1.2);
xline(0, 'k-', 'LineWidth', 0.8);
hold off
set(gca, 'YTick', 1:height(T), ...
    'YTickLabel', cellstr(strrep(T.case_name, "_", " ")), ...
    'TickLabelInterpreter', 'none');
grid on
xlabel('Delta d18O contribution at 50 km (per mil)');
title('Largest central-divide d18O shifts: contribution decomposition');
legend({'Mixing fraction', 'State 1 isotope', 'State 2 isotope', ...
    'Total delta'}, 'Location', 'best');
exportgraphics(fig, fullfile(controlsDir, ...
    'central_divide_top_case_contributions_50km.png'), 'Resolution', 240);
close(fig);
end

function azTable = buildAzimuthResponseTable(rows50)
A = rows50(rows50.group_name == "azimuth_fine", :);
azTable = table();
for i = 1:height(A)
    [stateName, deltaDeg] = parseAzimuthCase(A.case_name(i));
    row = table(stateName, deltaDeg, A.case_name(i), ...
        A.combined_d18O_permille(i), ...
        A.delta_combined_d18O_vs_baseline_permille(i), ...
        A.state1_precip_fraction(i), ...
        A.delta_state1_precip_fraction_vs_baseline(i), ...
        A.mixing_fraction_contribution_permille(i), ...
        A.state1_isotope_contribution_permille(i), ...
        A.state2_isotope_contribution_permille(i), ...
        'VariableNames', {'wind_state', 'azimuth_delta_deg', 'case_name', ...
        'combined_d18O_permille', ...
        'delta_combined_d18O_vs_baseline_permille', ...
        'state1_precip_fraction', ...
        'delta_state1_precip_fraction_vs_baseline', ...
        'mixing_fraction_contribution_permille', ...
        'state1_isotope_contribution_permille', ...
        'state2_isotope_contribution_permille'});
    azTable = [azTable; row]; %#ok<AGROW>
end
if isempty(azTable)
    return
end
[~, order] = sortrows([double(azTable.wind_state == "state2"), ...
    azTable.azimuth_delta_deg]);
azTable = azTable(order, :);
end

function az2Table = buildAz2TransitionResponseTable(rows50)
A = rows50(rows50.group_name == "az2_transition", :);
az2Table = table();
for i = 1:height(A)
    [stateName, deltaDeg] = parseAzimuthCase(A.case_name(i));
    row = table(stateName, deltaDeg, A.case_name(i), ...
        A.combined_d18O_permille(i), ...
        A.delta_combined_d18O_vs_baseline_permille(i), ...
        A.state1_precip_fraction(i), ...
        A.delta_state1_precip_fraction_vs_baseline(i), ...
        A.state2_d18O_permille(i), ...
        A.delta_state2_d18O_vs_baseline_permille(i), ...
        A.mixing_fraction_contribution_permille(i), ...
        A.state1_isotope_contribution_permille(i), ...
        A.state2_isotope_contribution_permille(i), ...
        'VariableNames', {'wind_state', 'azimuth_delta_deg', 'case_name', ...
        'combined_d18O_permille', ...
        'delta_combined_d18O_vs_baseline_permille', ...
        'state1_precip_fraction', ...
        'delta_state1_precip_fraction_vs_baseline', ...
        'state2_d18O_permille', 'delta_state2_d18O_vs_baseline_permille', ...
        'mixing_fraction_contribution_permille', ...
        'state1_isotope_contribution_permille', ...
        'state2_isotope_contribution_permille'});
    az2Table = [az2Table; row]; %#ok<AGROW>
end
if isempty(az2Table)
    return
end
[~, order] = sort(az2Table.azimuth_delta_deg);
az2Table = az2Table(order, :);
end

function [stateName, deltaDeg] = parseAzimuthCase(caseName)
caseName = string(caseName);
if startsWith(caseName, "Az1_")
    stateName = "state1";
    suffix = erase(caseName, "Az1_");
elseif startsWith(caseName, "Az2_")
    stateName = "state2";
    suffix = erase(caseName, "Az2_");
else
    error('Unexpected azimuth case name: %s', caseName);
end
if startsWith(suffix, "plus")
    signValue = 1;
    suffix = erase(suffix, "plus");
elseif startsWith(suffix, "minus")
    signValue = -1;
    suffix = erase(suffix, "minus");
else
    error('Unexpected azimuth case suffix: %s', suffix);
end
suffix = erase(suffix, "deg");
deltaDeg = signValue * str2double(suffix);
end

function plotAzimuthResponse(azTable, controlsDir)
if isempty(azTable)
    return
end
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 760]);
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

plotOneAzimuthPanel(azTable(azTable.wind_state == "state1", :), ...
    'State 1 azimuth perturbation');
plotOneAzimuthPanel(azTable(azTable.wind_state == "state2", :), ...
    'State 2 azimuth perturbation');

exportgraphics(fig, fullfile(controlsDir, ...
    'central_divide_azimuth_fine_response_50km.png'), 'Resolution', 240);
close(fig);
end

function plotOneAzimuthPanel(T, panelTitle)
[~, order] = sort(T.azimuth_delta_deg);
T = T(order, :);

nexttile
hold on
plot(T.azimuth_delta_deg, T.delta_combined_d18O_vs_baseline_permille, ...
    '-ok', 'LineWidth', 1.6, 'MarkerFaceColor', 'w');
plot(T.azimuth_delta_deg, T.mixing_fraction_contribution_permille, ...
    '-o', 'LineWidth', 1.4, 'Color', [0.20 0.45 0.70]);
plot(T.azimuth_delta_deg, T.state1_isotope_contribution_permille, ...
    '-o', 'LineWidth', 1.4, 'Color', [0.70 0.30 0.25]);
plot(T.azimuth_delta_deg, T.state2_isotope_contribution_permille, ...
    '-o', 'LineWidth', 1.4, 'Color', [0.25 0.55 0.35]);
yline(0, 'k-', 'LineWidth', 0.8);
hold off
grid on
xlabel('Azimuth perturbation (degrees)');
ylabel('Delta d18O at 50 km (per mil)');
title(panelTitle);
legend({'Total delta', 'Mixing fraction', 'State 1 isotope', ...
    'State 2 isotope'}, 'Location', 'best');
end

function plotAz2TransitionResponse(az2Table, controlsDir)
if isempty(az2Table)
    return
end

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 680]);
hold on
plot(az2Table.azimuth_delta_deg, ...
    az2Table.delta_combined_d18O_vs_baseline_permille, ...
    '-ok', 'LineWidth', 1.8, 'MarkerFaceColor', 'w');
plot(az2Table.azimuth_delta_deg, ...
    az2Table.mixing_fraction_contribution_permille, ...
    '-o', 'LineWidth', 1.5, 'Color', [0.20 0.45 0.70]);
plot(az2Table.azimuth_delta_deg, ...
    az2Table.state2_isotope_contribution_permille, ...
    '-o', 'LineWidth', 1.5, 'Color', [0.25 0.55 0.35]);
yline(0, 'k-', 'LineWidth', 0.8);
hold off
grid on
xlabel('State 2 azimuth perturbation (degrees)');
ylabel('Delta d18O at 50 km (per mil)');
title('State 2 azimuth transition response at central divide');
legend({'Total delta', 'Mixing fraction', 'State 2 isotope'}, ...
    'Location', 'best');
exportgraphics(fig, fullfile(controlsDir, ...
    'central_divide_az2_transition_response_50km.png'), 'Resolution', 240);
close(fig);
end
