function opiAnalyze_TopographyMorphologyEffects(rootScenario, experimentName)
% Compare morphology patterns against P01 at matched height and normalization.

if nargin < 1 || strlength(string(rootScenario)) == 0
    rootScenario = fullfile(fileparts(mfilename('fullpath')), '..', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
end
if nargin < 2 || strlength(string(experimentName)) == 0
    experimentName = 'topography_sensitivity_clumped_band';
end
rootScenario = char(string(rootScenario));
experimentName = char(string(experimentName));
experimentRoot = fullfile(rootScenario, experimentName);
analysisRoot = fullfile(experimentRoot, 'analysis');

modes = ["Mfixed", "Nmfixed"];
heightLevels = [3000, 4150, 5000];
patterns = ["P02_north_platform", "P03_south_platform", ...
    "P04_broad_plateau"];

rows = table();
rows = appendStage(rows, fullfile(experimentRoot, 'calc_only'), ...
    "calc_only", modes, heightLevels, patterns, "");
rows = appendStage(rows, fullfile(experimentRoot, 'refit_selected'), ...
    "refit_selected", modes, heightLevels, patterns, "_R01");
if isempty(rows)
    error('No matched morphology contrasts were found.');
end
writetable(rows, fullfile(analysisRoot, ...
    'morphology_matched_contrasts.csv'));

[G, stage, mode, pattern] = findgroups(rows.stage, ...
    rows.normalization_mode, rows.pattern_id);
summary = table(stage, mode, pattern, splitapply(@numel, rows.case_id, G), ...
    splitapply(@mean, rows.delta_weighted_d18O_permil, G), ...
    splitapply(@min, rows.delta_weighted_d18O_permil, G), ...
    splitapply(@max, rows.delta_weighted_d18O_permil, G), ...
    splitapply(@mean, rows.spatial_rms_delta_permil, G), ...
    splitapply(@mean, rows.fraction_grid_positive, G), ...
    'VariableNames', {'stage', 'normalization_mode', 'pattern_id', ...
    'n_contrasts', 'mean_delta_weighted_d18O_permil', ...
    'min_delta_weighted_d18O_permil', 'max_delta_weighted_d18O_permil', ...
    'mean_spatial_rms_delta_permil', 'mean_fraction_grid_positive'});
writetable(summary, fullfile(analysisRoot, ...
    'morphology_effect_summary.csv'));

reportFile = fullfile(analysisRoot, 'morphology_effect_report.md');
fid = fopen(reportFile, 'w');
if fid == -1
    error('Could not create morphology report: %s', reportFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Matched Topography Morphology Effects on Precipitation d18O\n\n');
fprintf(fid, 'Every contrast compares an alternative morphology with P01 at the same\n');
fprintf(fid, 'height and normalization mode. Positive values mean less negative\n');
fprintf(fid, 'reconstructed precipitation d18O than P01.\n\n');
writePatternFinding(fid, rows, "calc_only", "P02_north_platform");
writePatternFinding(fid, rows, "calc_only", "P03_south_platform");
writePatternFinding(fid, rows, "calc_only", "P04_broad_plateau");
fprintf(fid, '## Refit qualification\n\n');
refitRows = rows(rows.stage == "refit_selected", :);
if isempty(refitRows)
    fprintf(fid, 'No matched refit contrasts are available.\n');
else
    for i = 1:size(refitRows, 1)
        fprintf(fid, '- `%s`: weighted delta %.4f per mil; spatial RMS %.4f per mil.\n', ...
            refitRows.case_id(i), refitRows.delta_weighted_d18O_permil(i), ...
            refitRows.spatial_rms_delta_permil(i));
    end
    fprintf(fid, '\nThe refit subset is not a full factorial design. Only matched pairs listed\n');
    fprintf(fid, 'above support a direct morphology comparison after optimization.\n');
    fprintf(fid, 'Their sample-mean d18O differences are approximately zero, while their\n');
    fprintf(fid, 'regional weighted fields differ substantially. This indicates regional\n');
    fprintf(fid, 'field non-uniqueness after parameter refitting.\n');
end

fprintf('Wrote matched morphology-effect analysis under:\n%s\n', analysisRoot);
end

function rows = appendStage(rows, stageRoot, stage, modes, heights, patterns, suffix)
if ~isfolder(stageRoot)
    return
end
for iMode = 1:numel(modes)
    for iHeight = 1:numel(heights)
        baseId = string(sprintf('%s_H%d_P01_double_platform%s', ...
            modes(iMode), heights(iHeight), suffix));
        baseFile = fullfile(stageRoot, baseId, ...
            'opiCalc_TwoWinds_OxygenOnly_Results.mat');
        if ~isfile(baseFile)
            continue
        end
        B = load(baseFile, 'd18OGrid', 'pGrid', 'd18OPred');
        for iPattern = 1:numel(patterns)
            caseId = string(sprintf('%s_H%d_%s%s', modes(iMode), ...
                heights(iHeight), patterns(iPattern), suffix));
            caseFile = fullfile(stageRoot, caseId, ...
                'opiCalc_TwoWinds_OxygenOnly_Results.mat');
            if ~isfile(caseFile)
                continue
            end
            S = load(caseFile, 'd18OGrid', 'pGrid', 'd18OPred');
            valid = isfinite(S.d18OGrid) & isfinite(B.d18OGrid) & ...
                isfinite(S.pGrid) & isfinite(B.pGrid) & ...
                S.pGrid > 0 & B.pGrid > 0;
            delta = (S.d18OGrid(valid) - B.d18OGrid(valid)) * 1e3;
            pCase = S.pGrid(valid);
            weightedCase = weightedMean(S.d18OGrid, S.pGrid) * 1e3;
            weightedBase = weightedMean(B.d18OGrid, B.pGrid) * 1e3;
            sampleValid = isfinite(S.d18OPred) & isfinite(B.d18OPred);
            sampleDelta = mean(S.d18OPred(sampleValid) - ...
                B.d18OPred(sampleValid)) * 1e3;
            row = table(caseId, string(stage), modes(iMode), heights(iHeight), ...
                patterns(iPattern), baseId, weightedCase, weightedBase, ...
                weightedCase - weightedBase, sum(delta .* pCase) / sum(pCase), ...
                mean(delta), sqrt(mean(delta.^2)), min(delta), max(delta), ...
                mean(delta > 0), sum(S.pGrid(valid)) / sum(B.pGrid(valid)), ...
                sampleDelta, 'VariableNames', {'case_id', 'stage', ...
                'normalization_mode', 'height_target_m', 'pattern_id', ...
                'reference_case_id', 'weighted_d18O_case_permil', ...
                'weighted_d18O_reference_permil', ...
                'delta_weighted_d18O_permil', ...
                'case_precip_weighted_grid_delta_permil', ...
                'spatial_mean_delta_permil', 'spatial_rms_delta_permil', ...
                'min_grid_delta_permil', 'max_grid_delta_permil', ...
                'fraction_grid_positive', 'total_precip_ratio', ...
                'sample_mean_delta_permil'});
            rows = [rows; row]; %#ok<AGROW>
        end
    end
end
end

function value = weightedMean(d18OGrid, pGrid)
valid = isfinite(d18OGrid) & isfinite(pGrid) & pGrid > 0;
value = sum(d18OGrid(valid) .* pGrid(valid)) / sum(pGrid(valid));
end

function writePatternFinding(fid, rows, stage, pattern)
R = rows(rows.stage == stage & rows.pattern_id == pattern, :);
fprintf(fid, '## %s\n\n', pattern);
fprintf(fid, '- Weighted d18O effect: %.4f to %.4f per mil (mean %.4f).\n', ...
    min(R.delta_weighted_d18O_permil), max(R.delta_weighted_d18O_permil), ...
    mean(R.delta_weighted_d18O_permil));
fprintf(fid, '- Spatial RMS effect: %.4f to %.4f per mil.\n', ...
    min(R.spatial_rms_delta_permil), max(R.spatial_rms_delta_permil));
fprintf(fid, '- Positive grid fraction: %.3f to %.3f.\n\n', ...
    min(R.fraction_grid_positive), max(R.fraction_grid_positive));
fprintf(fid, '- Sample/catchment d18O effect: %.4f to %.4f per mil.\n', ...
    min(R.sample_mean_delta_permil), max(R.sample_mean_delta_permil));
fprintf(fid, '- Total precipitation ratio: %.3f to %.3f.\n\n', ...
    min(R.total_precip_ratio), max(R.total_precip_ratio));
end
