function opiAnalyze_QiangtangHeightRefinement(rootScenario, experimentName)
% Analyze the fine height matrix against meteoric-water d18O = -13.54 per mil.

if nargin < 1 || strlength(string(rootScenario)) == 0
    rootScenario = fullfile(fileparts(mfilename('fullpath')), '..', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
end
if nargin < 2 || strlength(string(experimentName)) == 0
    experimentName = 'topography_qiangtang_height_refinement';
end
target = -13.54;
analysisRoot = fullfile(char(string(rootScenario)), ...
    char(string(experimentName)), 'analysis');
summaryFile = fullfile(analysisRoot, ...
    'fixed_precipitation_d18O_summary.csv');
if ~isfile(summaryFile)
    error('Height-refinement summary not found: %s', summaryFile);
end
T = readtable(summaryFile, 'TextType', 'string');
[g, q, valley] = parseCases(T.case_id);
T.gangdese_target_m = g;
T.qiangtang_target_m = q;
T.valley_mode = valley;
T.target_d18O_permil = repmat(target, height(T), 1);
T.residual_50km_permil = T.weighted_d18O_50km_permil - target;
T.absolute_residual_50km_permil = abs(T.residual_50km_permil);
T = sortrows(T, {'gangdese_target_m', 'qiangtang_target_m', 'valley_mode'});
writetable(T, fullfile(analysisRoot, ...
    'height_refinement_target_comparison.csv'));

gValues = unique(g(valley == "Vnone"));
crossings = table();
for i = 1:numel(gValues)
    thisG = gValues(i);
    mask = T.gangdese_target_m == thisG & T.valley_mode == "Vnone";
    qValues = T.qiangtang_target_m(mask);
    dValues = T.weighted_d18O_50km_permil(mask);
    [qValues, order] = sort(qValues);
    dValues = dValues(order);
    [minResidual, bestIndex] = min(abs(dValues - target));
    crossing = interp1(dValues, qValues, target, 'linear', nan);
    row = table(thisG, qValues(bestIndex), dValues(bestIndex), ...
        dValues(bestIndex) - target, minResidual, crossing, ...
        'VariableNames', {'gangdese_target_m', 'closest_qiangtang_m', ...
        'closest_d18O_50km_permil', 'signed_residual_permil', ...
        'absolute_residual_permil', 'interpolated_qiangtang_m'});
    crossings = [crossings; row]; %#ok<AGROW>
end
writetable(crossings, fullfile(analysisRoot, ...
    'height_interpolated_estimates.csv'));

valleyEffects = table();
for i = 1:numel(gValues)
    thisG = gValues(i);
    noValley = T.gangdese_target_m == thisG & ...
        T.qiangtang_target_m == 4000 & T.valley_mode == "Vnone";
    lowValley = T.gangdese_target_m == thisG & ...
        T.qiangtang_target_m == 4000 & T.valley_mode == "V1500";
    if nnz(noValley) == 1 && nnz(lowValley) == 1
        dNo = T.weighted_d18O_50km_permil(noValley);
        dLow = T.weighted_d18O_50km_permil(lowValley);
        row = table(thisG, dNo, dLow, dLow - dNo, ...
            'VariableNames', {'gangdese_target_m', 'd18O_no_valley_permil', ...
            'd18O_1500m_valley_permil', 'valley_effect_permil'});
        valleyEffects = [valleyEffects; row]; %#ok<AGROW>
    end
end
writetable(valleyEffects, fullfile(analysisRoot, ...
    'valley_effects_50km.csv'));
writeReport(analysisRoot, target, crossings, valleyEffects);
fprintf('Wrote Qiangtang height-refinement analysis under:\n%s\n', analysisRoot);
end

function [g, q, valley] = parseCases(caseIds)
g = nan(size(caseIds));
q = nan(size(caseIds));
valley = strings(size(caseIds));
for i = 1:numel(caseIds)
    tokens = regexp(caseIds(i), '^G(\d+)_Q(\d+)_(V\w+)$', ...
        'tokens', 'once');
    if isempty(tokens)
        error('Unexpected refinement case ID: %s', caseIds(i));
    end
    g(i) = str2double(tokens{1});
    q(i) = str2double(tokens{2});
    valley(i) = string(tokens{3});
end
end

function writeReport(analysisRoot, target, crossings, valleyEffects)
fid = fopen(fullfile(analysisRoot, 'height_refinement_report.md'), 'w');
if fid == -1
    error('Could not create height-refinement report.');
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Qiangtang Height Refinement\n\n');
fprintf(fid, 'Independent meteoric-water target: **%.2f per mil**.\n\n', target);
fprintf(fid, 'Primary spatial support: precipitation-weighted 50 km radius around the proxy coordinate. Parameters are fixed at the joint oxygen-plus-clumped baseline solution.\n\n');
fprintf(fid, '## Height estimates without an imposed low valley\n\n');
fprintf(fid, '| Gangdese | Closest sampled Qiangtang | Modeled d18O | Linear crossing |\n');
fprintf(fid, '|---:|---:|---:|---:|\n');
for i = 1:height(crossings)
    fprintf(fid, '| %.0f m | %.0f m | %.3f per mil | %.0f m |\n', ...
        crossings.gangdese_target_m(i), crossings.closest_qiangtang_m(i), ...
        crossings.closest_d18O_50km_permil(i), ...
        crossings.interpolated_qiangtang_m(i));
end
fprintf(fid, '\nBoth allowed Gangdese end members place Qiangtang near 4.0-4.1 km.\n\n');
fprintf(fid, '## 1500 m valley effect at Qiangtang = 4000 m\n\n');
fprintf(fid, '| Gangdese | No valley | 1500 m valley | Difference |\n');
fprintf(fid, '|---:|---:|---:|---:|\n');
for i = 1:height(valleyEffects)
    fprintf(fid, '| %.0f m | %.3f | %.3f | %.3f per mil |\n', ...
        valleyEffects.gangdese_target_m(i), ...
        valleyEffects.d18O_no_valley_permil(i), ...
        valleyEffects.d18O_1500m_valley_permil(i), ...
        valleyEffects.valley_effect_permil(i));
end
fprintf(fid, '\nThe low-valley cases are substantially more negative at the proxy scale and do not match the target with the fixed baseline parameters.\n');
end
