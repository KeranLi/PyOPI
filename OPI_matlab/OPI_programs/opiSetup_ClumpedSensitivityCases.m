function opiSetup_ClumpedSensitivityCases(caseName, groupName)
% opiSetup_ClumpedSensitivityCases creates self-contained sensitivity cases
% from the clumped-temperature baseline scenario.
%
% Usage
%   opiSetup_ClumpedSensitivityCases
%       Create all first-batch case skeletons.
%
%   opiSetup_ClumpedSensitivityCases("T0_1_290K")
%       Create only the named case skeleton.
%
%   opiSetup_ClumpedSensitivityCases("", "parameter")
%       Create all first-batch local-parameter skeletons only.
%
%   opiSetup_ClumpedSensitivityCases("", "divide")
%       Create all first-batch divide-shift skeletons only.
%
%   opiSetup_ClumpedSensitivityCases("", "proxy")
%       Create all first-batch proxy/clumped skeletons only.
%
%   opiSetup_ClumpedSensitivityCases("", "mechanism")
%       Create second-batch mechanism-oriented local sensitivity skeletons.
%
%   opiSetup_ClumpedSensitivityCases("", "azimuth_fine")
%       Create calc-only fine wind-azimuth sensitivity skeletons.
%
%   opiSetup_ClumpedSensitivityCases("", "az2_transition")
%       Create calc-only State 2 azimuth transition skeletons.
%
%   opiSetup_ClumpedSensitivityCases("", "divide_calc_only")
%       Create calc-only divide-shift sensitivity skeletons.

if nargin < 1
    caseName = "";
end
if nargin < 2
    groupName = "";
end
caseName = string(caseName);
groupName = lower(string(groupName));

rootScenario = fullfile(fileparts(mfilename('fullpath')), '..', 'scenarios', ...
    'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
rootScenario = char(string(rootScenario));

baselineDir = fullfile(rootScenario, 'oxygen_clumped_ultra_aggressive');
baselineRun = fullfile(baselineDir, ...
    'Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive.run');
baselineBestRun = fullfile(baselineDir, ...
    'Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive_Best.run');

requiredFiles = {
    fullfile(rootScenario, 'Tibet_Eocene_30Ma_topo.mat')
    fullfile(rootScenario, 'Tibet_Eocene_30Ma_samples.xlsx')
    fullfile(rootScenario, 'proxy_clumped', 'clumped_temperature.xlsx')
    };
for i = 1:numel(requiredFiles)
    if ~isfile(requiredFiles{i})
        error('Required input file not found: %s', requiredFiles{i});
    end
end
if ~isfile(baselineRun)
    error('Baseline run file not found: %s', baselineRun);
end
if ~isfile(baselineBestRun)
    error('Baseline best-run file not found: %s', baselineBestRun);
end
sectionOrigin = defaultSectionOrigin();

parameterCases = buildParameterCases();
mechanismCases = buildMechanismCases();
azimuthFineCases = buildAzimuthFineCases();
az2TransitionCases = buildAz2TransitionCases();
divideCases = buildDivideCases();
divideCalcOnlyCases = buildDivideCases();
proxyCases = buildProxyCases();

[parameterCases, mechanismCases, azimuthFineCases, az2TransitionCases, ...
    divideCases, divideCalcOnlyCases, proxyCases] = filterGroups(parameterCases, ...
    mechanismCases, azimuthFineCases, az2TransitionCases, divideCases, ...
    divideCalcOnlyCases, proxyCases, groupName);

parameterCases = filterCases(parameterCases, caseName);
mechanismCases = filterCases(mechanismCases, caseName);
azimuthFineCases = filterCases(azimuthFineCases, caseName);
az2TransitionCases = filterCases(az2TransitionCases, caseName);
divideCases = filterCases(divideCases, caseName);
divideCalcOnlyCases = filterCases(divideCalcOnlyCases, caseName);
proxyCases = filterCases(proxyCases, caseName);

if ~all(groupName == "") ...
        && ~any(groupName == ["parameter", "mechanism", "azimuth_fine", ...
        "az2_transition", "divide", "divide_calc_only", "proxy"])
    error('Unknown group name: %s', groupName);
end
if ~all(caseName == "") && isempty(parameterCases) ...
        && isempty(mechanismCases) && isempty(azimuthFineCases) ...
        && isempty(az2TransitionCases) && isempty(divideCases) ...
        && isempty(divideCalcOnlyCases) && isempty(proxyCases)
    error('Requested case name not found in sensitivity definitions: %s', caseName);
end

for i = 1:numel(parameterCases)
    createParameterCase(rootScenario, baselineRun, baselineBestRun, ...
        parameterCases(i), sectionOrigin);
end
baselineBeta = readBestFitBeta(baselineBestRun);
for i = 1:numel(mechanismCases)
    createMechanismCase(rootScenario, baselineRun, baselineBestRun, ...
        baselineBeta, mechanismCases(i), sectionOrigin);
end
for i = 1:numel(azimuthFineCases)
    createMechanismLikeCase(rootScenario, baselineRun, baselineBestRun, ...
        baselineBeta, azimuthFineCases(i), sectionOrigin, ...
        'sensitivity_azimuth_fine_clumped', ...
        'Tibet_Eocene_30Ma_OxygenClumped_AzimuthFine_', ...
        "Azimuth fine local sensitivity");
end
for i = 1:numel(az2TransitionCases)
    createMechanismLikeCase(rootScenario, baselineRun, baselineBestRun, ...
        baselineBeta, az2TransitionCases(i), sectionOrigin, ...
        'sensitivity_az2_transition_clumped', ...
        'Tibet_Eocene_30Ma_OxygenClumped_Az2Transition_', ...
        "Az2 transition local sensitivity");
end
for i = 1:numel(divideCases)
    createDivideCase(rootScenario, baselineRun, divideCases(i), sectionOrigin);
end
for i = 1:numel(divideCalcOnlyCases)
    createDivideCalcOnlyCase(rootScenario, baselineRun, baselineBestRun, ...
        divideCalcOnlyCases(i), sectionOrigin);
end
for i = 1:numel(proxyCases)
    createProxyCase(rootScenario, baselineRun, proxyCases(i), sectionOrigin);
end

fprintf('Created clumped sensitivity case skeletons.\n');
fprintf('Parameter cases: %d\n', numel(parameterCases));
fprintf('Mechanism cases: %d\n', numel(mechanismCases));
fprintf('Azimuth fine cases: %d\n', numel(azimuthFineCases));
fprintf('Az2 transition cases: %d\n', numel(az2TransitionCases));
fprintf('Divide-shift cases: %d\n', numel(divideCases));
fprintf('Divide calc-only cases: %d\n', numel(divideCalcOnlyCases));
fprintf('Proxy/clumped cases: %d\n', numel(proxyCases));
end

function cases = filterCases(cases, caseName)
if isempty(cases) || caseName == ""
    return
end
names = string({cases.caseName});
cases = cases(names == caseName);
end

function [parameterCases, mechanismCases, azimuthFineCases, az2TransitionCases, ...
    divideCases, divideCalcOnlyCases, proxyCases] = filterGroups(parameterCases, ...
    mechanismCases, azimuthFineCases, az2TransitionCases, divideCases, ...
    divideCalcOnlyCases, proxyCases, groupName)
if groupName == ""
    return
end
switch groupName
    case "parameter"
        mechanismCases = struct.empty(0, 1);
        azimuthFineCases = struct.empty(0, 1);
        az2TransitionCases = struct.empty(0, 1);
        divideCases = struct.empty(0, 1);
        divideCalcOnlyCases = struct.empty(0, 1);
        proxyCases = struct.empty(0, 1);
    case "mechanism"
        parameterCases = struct.empty(0, 1);
        azimuthFineCases = struct.empty(0, 1);
        az2TransitionCases = struct.empty(0, 1);
        divideCases = struct.empty(0, 1);
        divideCalcOnlyCases = struct.empty(0, 1);
        proxyCases = struct.empty(0, 1);
    case "azimuth_fine"
        parameterCases = struct.empty(0, 1);
        mechanismCases = struct.empty(0, 1);
        az2TransitionCases = struct.empty(0, 1);
        divideCases = struct.empty(0, 1);
        divideCalcOnlyCases = struct.empty(0, 1);
        proxyCases = struct.empty(0, 1);
    case "az2_transition"
        parameterCases = struct.empty(0, 1);
        mechanismCases = struct.empty(0, 1);
        azimuthFineCases = struct.empty(0, 1);
        divideCases = struct.empty(0, 1);
        divideCalcOnlyCases = struct.empty(0, 1);
        proxyCases = struct.empty(0, 1);
    case "divide"
        parameterCases = struct.empty(0, 1);
        mechanismCases = struct.empty(0, 1);
        azimuthFineCases = struct.empty(0, 1);
        az2TransitionCases = struct.empty(0, 1);
        divideCalcOnlyCases = struct.empty(0, 1);
        proxyCases = struct.empty(0, 1);
    case "divide_calc_only"
        parameterCases = struct.empty(0, 1);
        mechanismCases = struct.empty(0, 1);
        azimuthFineCases = struct.empty(0, 1);
        az2TransitionCases = struct.empty(0, 1);
        divideCases = struct.empty(0, 1);
        proxyCases = struct.empty(0, 1);
    case "proxy"
        parameterCases = struct.empty(0, 1);
        mechanismCases = struct.empty(0, 1);
        azimuthFineCases = struct.empty(0, 1);
        az2TransitionCases = struct.empty(0, 1);
        divideCases = struct.empty(0, 1);
        divideCalcOnlyCases = struct.empty(0, 1);
end
end

function cases = buildParameterCases()
cases = [
    makeParameterCase("T0_1_285K", 3, 285, "T0_1 = 285 K")
    makeParameterCase("T0_1_287p5K", 3, 287.5, "T0_1 = 287.5 K")
    makeParameterCase("T0_1_290K", 3, 290, "T0_1 = 290 K")
    makeParameterCase("T0_1_292p5K", 3, 292.5, "T0_1 = 292.5 K")
    makeParameterCase("T0_1_295K", 3, 295, "T0_1 = 295 K")
    makeParameterCase("T0_1_296p25K", 3, 296.25, "T0_1 = 296.25 K")
    makeParameterCase("T0_1_297p5K", 3, 297.5, "T0_1 = 297.5 K")
    makeParameterCase("T0_1_298p75K", 3, 298.75, "T0_1 = 298.75 K")
    makeParameterCase("T0_2_270K", 13, 270, "T0_2 = 270 K")
    makeParameterCase("T0_2_272p5K", 13, 272.5, "T0_2 = 272.5 K")
    makeParameterCase("T0_2_275K", 13, 275, "T0_2 = 275 K")
    makeParameterCase("T0_2_277p5K", 13, 277.5, "T0_2 = 277.5 K")
    makeParameterCase("T0_2_280K", 13, 280, "T0_2 = 280 K")
    makeParameterCase("M_1_0p30", 4, 0.30, "M_1 = 0.30")
    makeParameterCase("M_1_0p40", 4, 0.40, "M_1 = 0.40")
    makeParameterCase("M_1_0p50", 4, 0.50, "M_1 = 0.50")
    makeParameterCase("M_1_0p60", 4, 0.60, "M_1 = 0.60")
    makeParameterCase("M_1_0p70", 4, 0.70, "M_1 = 0.70")
    makeParameterCase("M_2_0p05", 14, 0.05, "M_2 = 0.05")
    makeParameterCase("M_2_0p10", 14, 0.10, "M_2 = 0.10")
    makeParameterCase("M_2_0p15", 14, 0.15, "M_2 = 0.15")
    makeParameterCase("M_2_0p20", 14, 0.20, "M_2 = 0.20")
    makeParameterCase("M_2_0p25", 14, 0.25, "M_2 = 0.25")
    makeParameterCase("fraction_0p50", 10, 0.50, "fraction = 0.50")
    makeParameterCase("fraction_0p60", 10, 0.60, "fraction = 0.60")
    makeParameterCase("fraction_0p70", 10, 0.70, "fraction = 0.70")
    makeParameterCase("fraction_0p80", 10, 0.80, "fraction = 0.80")
    makeParameterCase("fraction_0p90", 10, 0.90, "fraction = 0.90")
    ];
end

function cases = buildMechanismCases()
cases = [
    makeMechanismCase("d18O0_1_dm2permil", 7, "delta", -0.002, ...
    "State 1 source d18O0 -2 per mil")
    makeMechanismCase("d18O0_1_dp2permil", 7, "delta", 0.002, ...
    "State 1 source d18O0 +2 per mil")
    makeMechanismCase("d18O0_2_dm2permil", 17, "delta", -0.002, ...
    "State 2 source d18O0 -2 per mil")
    makeMechanismCase("d18O0_2_dp2permil", 17, "delta", 0.002, ...
    "State 2 source d18O0 +2 per mil")
    makeMechanismCase("d18Olat1_lessNeg0p5", 8, "delta", 0.0005, ...
    "State 1 d18O latitude gradient 0.5 per mil/deg less negative")
    makeMechanismCase("d18Olat1_moreNeg0p5", 8, "delta", -0.0005, ...
    "State 1 d18O latitude gradient 0.5 per mil/deg more negative")
    makeMechanismCase("d18Olat2_lessNeg0p5", 18, "delta", 0.0005, ...
    "State 2 d18O latitude gradient 0.5 per mil/deg less negative")
    makeMechanismCase("d18Olat2_moreNeg0p5", 18, "delta", -0.0005, ...
    "State 2 d18O latitude gradient 0.5 per mil/deg more negative")
    makeMechanismCase("fP1_0p25", 9, "absolute", 0.25, ...
    "State 1 residual precipitation fraction fP = 0.25")
    makeMechanismCase("fP1_0p60", 9, "absolute", 0.60, ...
    "State 1 residual precipitation fraction fP = 0.60")
    makeMechanismCase("fP2_0p25", 19, "absolute", 0.25, ...
    "State 2 residual precipitation fraction fP = 0.25")
    makeMechanismCase("fP2_0p60", 19, "absolute", 0.60, ...
    "State 2 residual precipitation fraction fP = 0.60")
    makeMechanismCase("Az1_minus10deg", 2, "delta", -10, ...
    "State 1 wind azimuth -10 deg")
    makeMechanismCase("Az1_plus10deg", 2, "delta", 10, ...
    "State 1 wind azimuth +10 deg")
    makeMechanismCase("Az2_minus10deg", 12, "delta", -10, ...
    "State 2 wind azimuth -10 deg")
    makeMechanismCase("Az2_plus10deg", 12, "delta", 10, ...
    "State 2 wind azimuth +10 deg")
    ];
end

function cases = buildAzimuthFineCases()
steps = [-15, -10, -5, 5, 10, 15];
cases = repmat(makeMechanismCase("placeholder", 2, "delta", 0, ""), 0, 1);
for i = 1:numel(steps)
    suffix = formatSignedDeg(steps(i));
    cases(end+1, 1) = makeMechanismCase("Az1_" + suffix, 2, "delta", ...
        steps(i), "State 1 wind azimuth " + sprintf('%+d deg', steps(i))); %#ok<AGROW>
end
for i = 1:numel(steps)
    suffix = formatSignedDeg(steps(i));
    cases(end+1, 1) = makeMechanismCase("Az2_" + suffix, 12, "delta", ...
        steps(i), "State 2 wind azimuth " + sprintf('%+d deg', steps(i))); %#ok<AGROW>
end
end

function cases = buildAz2TransitionCases()
steps = 2:2:18;
cases = repmat(makeMechanismCase("placeholder", 12, "delta", 0, ""), 0, 1);
for i = 1:numel(steps)
    suffix = formatSignedDeg(steps(i));
    cases(end+1, 1) = makeMechanismCase("Az2_" + suffix, 12, "delta", ...
        steps(i), "State 2 wind azimuth " + sprintf('%+d deg', steps(i))); %#ok<AGROW>
end
end

function cases = buildDivideCases()
cases = [
    makeDivideCase("north_030deg", "Tibet_Eocene_30Ma_topo_divide_main_north030.mat")
    makeDivideCase("north_020deg", "Tibet_Eocene_30Ma_topo_divide_main_north020.mat")
    makeDivideCase("north_010deg", "Tibet_Eocene_30Ma_topo_divide_main_north010.mat")
    makeDivideCase("base_000deg", "Tibet_Eocene_30Ma_topo_divide_main.mat")
    makeDivideCase("south_010deg", "Tibet_Eocene_30Ma_topo_divide_main_south010.mat")
    makeDivideCase("south_020deg", "Tibet_Eocene_30Ma_topo_divide_main_south020.mat")
    makeDivideCase("south_030deg", "Tibet_Eocene_30Ma_topo_divide_main_south030.mat")
    ];
end

function cases = buildProxyCases()
cases = [
    makeProxyCase("offset0_warmest", 0, 4, "warmest")
    makeProxyCase("offset5_warmest", 5, 4, "warmest")
    makeProxyCase("offset7_warmest", 7, 4, "warmest")
    makeProxyCase("offset10_warmest", 10, 4, "warmest")
    makeProxyCase("offset7_annual", 7, 4, "annual")
    makeProxyCase("offset7_jja", 7, 4, "jja")
    ];
end

function caseDef = makeParameterCase(caseName, betaIndex, betaValue, label)
caseDef = struct('caseName', caseName, 'betaIndex', betaIndex, ...
    'betaValue', betaValue, 'label', label);
end

function caseDef = makeMechanismCase(caseName, betaIndex, valueMode, value, label)
caseDef = struct('caseName', caseName, 'betaIndex', betaIndex, ...
    'valueMode', string(valueMode), 'value', value, 'label', label);
end

function caseDef = makeDivideCase(caseName, divideFileSource)
caseDef = struct('caseName', caseName, 'divideFileSource', divideFileSource);
end

function caseDef = makeProxyCase(caseName, offsetC, sigmaC, season)
caseDef = struct('caseName', caseName, 'offsetC', offsetC, ...
    'sigmaC', sigmaC, 'season', string(season));
end

function createParameterCase(rootScenario, baselineRun, baselineBestRun, ...
    caseDef, sectionOrigin)
groupDir = fullfile(rootScenario, 'sensitivity_parameter_local_clumped');
caseDir = fullfile(groupDir, caseDef.caseName);
ensureEmptyCaseDir(caseDir);
copyStaticScenarioInputs(rootScenario, caseDir, 'Tibet_Eocene_30Ma_topo_divide_main.mat');

runName = ['Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive_', char(caseDef.caseName), '.run'];
bestRunName = ['Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive_', char(caseDef.caseName), '_Best.run'];
runOut = fullfile(caseDir, runName);
bestRunOut = fullfile(caseDir, bestRunName);

writeRunFileForCase(baselineRun, runOut, caseDir, ...
    "Local parameter sensitivity: " + caseDef.caseName, sectionOrigin);
writeBestRunFileForCase(baselineBestRun, bestRunOut, caseDir, ...
    caseDef.betaIndex, caseDef.betaValue, ...
    "Local parameter sensitivity: " + caseDef.caseName, sectionOrigin);
writeCaseReadme(caseDir, [runName, newline, bestRunName], ...
    sprintf(['Case type: local parameter sensitivity\n' ...
    'Modified beta index: %d\n' ...
    'Modified beta value: %.8g\n' ...
    'Fixed section origin: %.5f, %.5f\n' ...
    'Run mode: no refit, rerun opiCalc and downstream diagnostics only.\n'], ...
    caseDef.betaIndex, caseDef.betaValue, sectionOrigin(1), sectionOrigin(2)));
end

function createMechanismCase(rootScenario, baselineRun, baselineBestRun, ...
    baselineBeta, caseDef, sectionOrigin)
createMechanismLikeCase(rootScenario, baselineRun, baselineBestRun, ...
    baselineBeta, caseDef, sectionOrigin, ...
    'sensitivity_mechanism_local_clumped', ...
    'Tibet_Eocene_30Ma_OxygenClumped_Mechanism_', ...
    "Mechanism local sensitivity");
end

function createMechanismLikeCase(rootScenario, baselineRun, baselineBestRun, ...
    baselineBeta, caseDef, sectionOrigin, groupFolder, runPrefix, titlePrefix)
groupDir = fullfile(rootScenario, groupFolder);
caseDir = fullfile(groupDir, caseDef.caseName);
ensureEmptyCaseDir(caseDir);
copyStaticScenarioInputs(rootScenario, caseDir, 'Tibet_Eocene_30Ma_topo_divide_main.mat');

runName = [runPrefix, char(caseDef.caseName), '.run'];
bestRunName = [runPrefix, char(caseDef.caseName), '_Best.run'];
runOut = fullfile(caseDir, runName);
bestRunOut = fullfile(caseDir, bestRunName);

if caseDef.valueMode == "delta"
    betaValue = baselineBeta(caseDef.betaIndex) + caseDef.value;
else
    betaValue = caseDef.value;
end

writeRunFileForCase(baselineRun, runOut, caseDir, ...
    titlePrefix + ": " + caseDef.caseName, sectionOrigin);
writeBestRunFileForCase(baselineBestRun, bestRunOut, caseDir, ...
    caseDef.betaIndex, betaValue, ...
    titlePrefix + ": " + caseDef.caseName, sectionOrigin);
writeCaseReadme(caseDir, [runName, newline, bestRunName], ...
    sprintf(['Case type: %s\n' ...
    'Mechanism label: %s\n' ...
    'Modified beta index: %d\n' ...
    'Value mode: %s\n' ...
    'Input value: %.8g\n' ...
    'Final beta value: %.8g\n' ...
    'Fixed section origin: %.5f, %.5f\n' ...
    'Run mode: no refit, rerun opiCalc and downstream diagnostics only.\n'], ...
    titlePrefix, caseDef.label, caseDef.betaIndex, caseDef.valueMode, ...
    caseDef.value, betaValue, sectionOrigin(1), sectionOrigin(2)));
end

function createDivideCase(rootScenario, baselineRun, caseDef, sectionOrigin)
groupDir = fullfile(rootScenario, 'sensitivity_divide_shift_clumped');
caseDir = fullfile(groupDir, caseDef.caseName);
ensureEmptyCaseDir(caseDir);
copyStaticScenarioInputs(rootScenario, caseDir, caseDef.divideFileSource);

runName = ['Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive_', char(caseDef.caseName), '.run'];
runOut = fullfile(caseDir, runName);
writeRunFileForCase(baselineRun, runOut, caseDir, ...
    "Divide shift sensitivity: " + caseDef.caseName, sectionOrigin);
writeCaseReadme(caseDir, runName, sprintf(['Case type: divide-shift sensitivity\n' ...
    'Divide source: %s\n' ...
    'Fixed section origin: %.5f, %.5f\n' ...
    'Run mode: rerun clumped fit, write new best-run file, rerun opiCalc and maps.\n'], ...
    caseDef.divideFileSource, sectionOrigin(1), sectionOrigin(2)));
end

function createDivideCalcOnlyCase(rootScenario, baselineRun, baselineBestRun, ...
    caseDef, sectionOrigin)
groupDir = fullfile(rootScenario, 'sensitivity_divide_calc_only_clumped');
caseDir = fullfile(groupDir, caseDef.caseName);
ensureEmptyCaseDir(caseDir);
copyStaticScenarioInputs(rootScenario, caseDir, caseDef.divideFileSource);

runName = ['Tibet_Eocene_30Ma_OxygenClumped_DivideCalcOnly_', char(caseDef.caseName), '.run'];
bestRunName = ['Tibet_Eocene_30Ma_OxygenClumped_DivideCalcOnly_', char(caseDef.caseName), '_Best.run'];
runOut = fullfile(caseDir, runName);
bestRunOut = fullfile(caseDir, bestRunName);
writeRunFileForCase(baselineRun, runOut, caseDir, ...
    "Divide calc-only sensitivity: " + caseDef.caseName, sectionOrigin);
writeBestRunFileForCaseWithoutBetaChange(baselineBestRun, bestRunOut, caseDir, ...
    "Divide calc-only sensitivity: " + caseDef.caseName, sectionOrigin);
writeCaseReadme(caseDir, [runName, newline, bestRunName], ...
    sprintf(['Case type: divide calc-only sensitivity\n' ...
    'Divide source: %s\n' ...
    'Fixed section origin: %.5f, %.5f\n' ...
    'Run mode: no refit; fixed baseline best-fit beta; rerun opiCalc and downstream diagnostics only.\n'], ...
    caseDef.divideFileSource, sectionOrigin(1), sectionOrigin(2)));
end

function createProxyCase(rootScenario, baselineRun, caseDef, sectionOrigin)
groupDir = fullfile(rootScenario, 'sensitivity_proxy_clumped');
caseDir = fullfile(groupDir, caseDef.caseName);
ensureEmptyCaseDir(caseDir);
copyStaticScenarioInputs(rootScenario, caseDir, 'Tibet_Eocene_30Ma_topo_divide_main.mat');

runName = ['Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive_', char(caseDef.caseName), '.run'];
runOut = fullfile(caseDir, runName);
writeRunFileForCase(baselineRun, runOut, caseDir, ...
    "Proxy/clumped sensitivity: " + caseDef.caseName, sectionOrigin);
writeProxyConfig(caseDir, caseDef.offsetC, caseDef.sigmaC, caseDef.season);
writeCaseReadme(caseDir, runName, sprintf(['Case type: proxy/clumped sensitivity\n' ...
    'dolomiteOffsetC: %.8g\n' ...
    'sigmaOffsetC: %.8g\n' ...
    'clumpedSeason: %s\n' ...
    'Fixed section origin: %.5f, %.5f\n' ...
    'Run mode: rerun clumped fit, write new best-run file, rerun opiCalc and maps.\n'], ...
    caseDef.offsetC, caseDef.sigmaC, caseDef.season, ...
    sectionOrigin(1), sectionOrigin(2)));
end

function ensureEmptyCaseDir(caseDir)
if isfolder(caseDir)
    warning('Case directory already exists, leaving existing contents in place: %s', caseDir);
else
    mkdir(caseDir);
end
end

function copyStaticScenarioInputs(rootScenario, caseDir, divideFileSource)
copyfile(fullfile(rootScenario, 'Tibet_Eocene_30Ma_topo.mat'), ...
    fullfile(caseDir, 'Tibet_Eocene_30Ma_topo.mat'));
copyfile(fullfile(rootScenario, 'Tibet_Eocene_30Ma_samples.xlsx'), ...
    fullfile(caseDir, 'Tibet_Eocene_30Ma_samples.xlsx'));
copyDivideFile(rootScenario, caseDir, divideFileSource);
copyProxyInputs(rootScenario, caseDir);
end

function copyDivideFile(rootScenario, caseDir, divideFileSource)
source = fullfile(rootScenario, divideFileSource);
target = fullfile(caseDir, 'Tibet_Eocene_30Ma_topo_divide_main.mat');
if ~isfile(source)
    error('Divide file not found: %s', source);
end
copyfile(source, target);
end

function copyProxyInputs(rootScenario, caseDir)
proxyDir = fullfile(caseDir, 'proxy_clumped');
if ~isfolder(proxyDir)
    mkdir(proxyDir);
end
copyfile(fullfile(rootScenario, 'proxy_clumped', 'clumped_temperature.xlsx'), ...
    fullfile(proxyDir, 'clumped_temperature.xlsx'));
if isfile(fullfile(rootScenario, 'proxy_clumped', 'clumped_temperature.csv'))
    copyfile(fullfile(rootScenario, 'proxy_clumped', 'clumped_temperature.csv'), ...
        fullfile(proxyDir, 'clumped_temperature.csv'));
end
end

function beta = readBestFitBeta(bestRun)
lines = readlines(bestRun, 'WhitespaceRule', 'preserve');
idx = findActiveLineIndices(lines);
beta = str2num(lines(idx(end))); %#ok<ST2NM>
if isempty(beta)
    error('Could not parse best-fit beta from: %s', bestRun);
end
end

function writeRunFileForCase(sourceRun, targetRun, caseDir, caseTitle, sectionOrigin)
lines = readlines(sourceRun, 'WhitespaceRule', 'preserve');
idx = findActiveLineIndices(lines);
lines(idx(1)) = caseTitle;
lines(idx(3)) = string(caseDir);
lines(idx(4)) = "no";
lines(idx(10)) = formatLonLatLine(sectionOrigin);
writelines(lines, targetRun);
end

function writeBestRunFileForCase(sourceRun, targetRun, caseDir, betaIndex, ...
    betaValue, caseTitle, sectionOrigin)
lines = readlines(sourceRun, 'WhitespaceRule', 'preserve');
idx = findActiveLineIndices(lines);
lines(idx(1)) = caseTitle;
lines(idx(3)) = string(caseDir);
lines(idx(4)) = "no";
lines(idx(10)) = formatLonLatLine(sectionOrigin);
betaLine = idx(end);
beta = str2num(lines(betaLine)); %#ok<ST2NM>
if isempty(beta)
    error('Could not parse best-fit beta from: %s', sourceRun);
end
beta(betaIndex) = betaValue;
lines(betaLine) = join(compose('%.8g', beta), ', ');
writelines(lines, targetRun);
end

function writeBestRunFileForCaseWithoutBetaChange(sourceRun, targetRun, ...
    caseDir, caseTitle, sectionOrigin)
lines = readlines(sourceRun, 'WhitespaceRule', 'preserve');
idx = findActiveLineIndices(lines);
lines(idx(1)) = caseTitle;
lines(idx(3)) = string(caseDir);
lines(idx(4)) = "no";
lines(idx(10)) = formatLonLatLine(sectionOrigin);
writelines(lines, targetRun);
end

function writeProxyConfig(caseDir, offsetC, sigmaC, season)
configFile = fullfile(caseDir, 'proxy_clumped', 'clumped_fit_config.csv');
fid = fopen(configFile, 'w');
if fid == -1
    error('Could not create proxy config: %s', configFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'dolomiteOffsetC,sigmaOffsetC,clumpedSeason\n');
fprintf(fid, '%.8g,%.8g,%s\n', offsetC, sigmaC, season);
end

function writeCaseReadme(caseDir, runFilesText, detailText)
readmeFile = fullfile(caseDir, 'README.txt');
fid = fopen(readmeFile, 'w');
if fid == -1
    error('Could not create README: %s', readmeFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'Self-contained OPI sensitivity case\n\n');
fprintf(fid, 'Run files:\n%s\n\n', runFilesText);
fprintf(fid, '%s', detailText);
end

function idx = findActiveLineIndices(lines)
idx = [];
for i = 1:numel(lines)
    str = strip(string(lines(i)));
    if strlength(str) == 0
        continue
    end
    if startsWith(str, "%")
        continue
    end
    idx(end+1) = i; %#ok<AGROW>
end
if numel(idx) < 16
    error('Run file format did not match expected active-line count.');
end
end

function sectionOrigin = defaultSectionOrigin()
sectionOrigin = [87.2, 32.9];
end

function text = formatLonLatLine(sectionOrigin)
text = sprintf('%.5f, %.5f', sectionOrigin(1), sectionOrigin(2));
end

function text = formatSignedDeg(value)
if value < 0
    text = "minus" + sprintf('%02ddeg', abs(value));
else
    text = "plus" + sprintf('%02ddeg', value);
end
end
