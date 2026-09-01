function opiAnalyze_ValleyHeightSensitivity(rootScenario, experimentName)
% Analyze intermontane valley elevation against d18O = -13.54 per mil.

if nargin < 1 || strlength(string(rootScenario)) == 0
    rootScenario = fullfile(fileparts(mfilename('fullpath')), '..', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
end
if nargin < 2 || strlength(string(experimentName)) == 0
    experimentName = 'topography_valley_height_sensitivity';
end
target = -13.54;
experimentRoot = fullfile(char(string(rootScenario)), ...
    char(string(experimentName)));
analysisRoot = fullfile(experimentRoot, 'analysis');
T = readtable(fullfile(analysisRoot, ...
    'fixed_precipitation_d18O_summary.csv'), 'TextType', 'string');
Q = readtable(fullfile(experimentRoot, 'design', ...
    'topography_quality_control.csv'), 'TextType', 'string');
if height(T) ~= height(Q) || ~all(ismember(T.case_id, Q.case_id))
    error('Result and topography QC case sets do not match.');
end
[~, location] = ismember(T.case_id, Q.case_id);
T.valley_core_mean_m = Q.valley_core_mean_m(location);
T.gangdese_target_m = parseGangdese(T.case_id);
T.is_continuous_highland = endsWith(T.case_id, '_Vnone');
T.target_d18O_permil = repmat(target, height(T), 1);
T.residual_50km_permil = T.weighted_d18O_50km_permil - target;
T.absolute_residual_50km_permil = abs(T.residual_50km_permil);
T = sortrows(T, {'gangdese_target_m', 'valley_core_mean_m'});
writetable(T, fullfile(analysisRoot, 'valley_target_comparison.csv'));

gValues = unique(T.gangdese_target_m);
R = table();
for i = 1:numel(gValues)
    g = gValues(i);
    mask = T.gangdese_target_m == g;
    valley = T.valley_core_mean_m(mask);
    d18O = T.weighted_d18O_50km_permil(mask);
    ids = T.case_id(mask);
    [valley, order] = sort(valley);
    d18O = d18O(order);
    ids = ids(order);
    [minimumResidual, best] = min(abs(d18O - target));
    crossing = interp1(d18O, valley, target, 'linear', nan);
    monotonic = all(diff(d18O) > 0);
    row = table(g, ids(best), valley(best), d18O(best), ...
        d18O(best) - target, minimumResidual, crossing, monotonic, ...
        d18O(end) - d18O(1), ...
        'VariableNames', {'gangdese_target_m', 'closest_case', ...
        'closest_effective_valley_m', 'closest_d18O_50km_permil', ...
        'signed_residual_permil', 'absolute_residual_permil', ...
        'interpolated_effective_valley_m', 'monotonic_response', ...
        'full_range_d18O_change_permil'});
    R = [R; row]; %#ok<AGROW>
end
writetable(R, fullfile(analysisRoot, 'valley_interpolated_estimates.csv'));
writeReport(analysisRoot, target, T, R);
fprintf('Wrote valley-height sensitivity analysis under:\n%s\n', analysisRoot);
end

function values = parseGangdese(caseIds)
values = nan(size(caseIds));
for i = 1:numel(caseIds)
    token = regexp(caseIds(i), '^G(\d+)_', 'tokens', 'once');
    if isempty(token)
        error('Unexpected valley case ID: %s', caseIds(i));
    end
    values(i) = str2double(token{1});
end
end

function writeReport(analysisRoot, target, T, R)
fid = fopen(fullfile(analysisRoot, 'valley_height_report.md'), 'w');
if fid == -1
    error('Could not create valley-height report.');
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Intermontane Valley Height Sensitivity\n\n');
fprintf(fid, 'Qiangtang is fixed at 4000 m. The independent 50 km precipitation-weighted meteoric-water target is **%.2f per mil**.\n\n', target);
for i = 1:height(R)
    g = R.gangdese_target_m(i);
    fprintf(fid, '## Gangdese %.0f m\n\n', g);
    fprintf(fid, '| Effective valley elevation | Geometry | 50 km d18O | Residual |\n');
    fprintf(fid, '|---:|---|---:|---:|\n');
    rows = T(T.gangdese_target_m == g, :);
    for j = 1:height(rows)
        geometry = "flat valley";
        if rows.is_continuous_highland(j)
            geometry = "continuous highland";
        end
        fprintf(fid, '| %.0f m | %s | %.3f | %+.3f |\n', ...
            rows.valley_core_mean_m(j), geometry, ...
            rows.weighted_d18O_50km_permil(j), ...
            rows.residual_50km_permil(j));
    end
    fprintf(fid, '\nThe response is monotonic: **%s**. The interpolated effective valley elevation at the target is **%.0f m**.\n\n', ...
        string(R.monotonic_response(i)), ...
        R.interpolated_effective_valley_m(i));
end
fprintf(fid, 'The final interpolation joins flat-floor cases to a sloping continuous-highland case, so it is an effective corridor elevation rather than a uniquely resolved flat valley floor.\n');
end
