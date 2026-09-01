function result = opiEstimate_Aridity(resultFile, varargin)
% opiEstimate_Aridity estimates aridity from a fitted OPI calculation.
%
% The function reads spatial fields from an
% opiCalc_TwoWinds_OxygenOnly_Results.mat file. It does not rerun or modify
% OPIfit. The primary output is a within-case relative aridity field. The
% absolute P/PET fields are scale scenarios, not calibrated annual
% paleoclimate reconstructions.
%
% Example:
%   result = opiEstimate_Aridity('opiCalc_TwoWinds_OxygenOnly_Results.mat');
%
% Name-value options:
%   OutputDir                  output directory (default: aridity_analysis)
%   PrecipitationScales       multipliers for annualized OPI precipitation
%                             (default: [0.01 0.03 0.1 0.3 1])
%   PETScale                  multiplier for Oudin PET (default: 1)
%   MinimumPrecipitationFraction
%                             wet-cell threshold relative to max P
%                             (default: 1e-6)
%   WriteOutputs              write MAT/CSV files (default: true)
%   MakeFigure                write PNG/FIG maps (default: true)

p = inputParser;
addRequired(p, 'resultFile', @(x) ischar(x) || isstring(x));
addParameter(p, 'OutputDir', "", @(x) ischar(x) || isstring(x));
addParameter(p, 'PrecipitationScales', [0.01, 0.03, 0.1, 0.3, 1], ...
    @(x) isnumeric(x) && isvector(x) && ~isempty(x) && ...
    all(isfinite(x)) && all(x > 0));
addParameter(p, 'PETScale', 1, ...
    @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
addParameter(p, 'MinimumPrecipitationFraction', 1e-6, ...
    @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0 && x < 1);
addParameter(p, 'WriteOutputs', true, ...
    @(x) islogical(x) && isscalar(x));
addParameter(p, 'MakeFigure', true, ...
    @(x) islogical(x) && isscalar(x));
parse(p, resultFile, varargin{:});
opts = p.Results;

resultFile = char(string(resultFile));
if ~isfile(resultFile)
    error('OPI result file not found: %s', resultFile);
end
[resultDir, ~, ~] = fileparts(resultFile);
outputDir = char(string(opts.OutputDir));
if isempty(outputDir)
    outputDir = fullfile(resultDir, 'aridity_analysis');
end
if (opts.WriteOutputs || opts.MakeFigure) && ~isfolder(outputDir)
    mkdir(outputDir);
end

requiredFields = {'lon', 'lat', 'pGrid', 'pGrid_1', 'pGrid_2', ...
    'T_1', 'T_2', 'gammaSat_1', 'gammaSat_2', 'topoFile', 'dataPath'};
requiredFields{end+1} = 'rTukey';
available = who('-file', resultFile);
missing = setdiff(requiredFields, available);
if ~isempty(missing)
    error('OPI result is missing required field(s): %s', ...
        strjoin(missing, ', '));
end
S = load(resultFile, requiredFields{:});

hGrid = loadTopography(S, resultDir);
expectedSize = [numel(S.lat), numel(S.lon)];
gridFields = {'pGrid', 'pGrid_1', 'pGrid_2'};
for i = 1:numel(gridFields)
    if ~isequal(size(S.(gridFields{i})), expectedSize)
        error('%s size does not match lat/lon dimensions.', gridFields{i});
    end
end
if ~isequal(size(hGrid), expectedSize)
    error('Topography size does not match lat/lon dimensions.');
end

TC2K = 273.15;
temperatureState1C = S.T_1(1) - S.gammaSat_1(1) .* hGrid - TC2K;
temperatureState2C = S.T_2(1) - S.gammaSat_2(1) .* hGrid - TC2K;
temperatureCombinedC = ...
    (S.pGrid_1 .* temperatureState1C + ...
    S.pGrid_2 .* temperatureState2C) ./ S.pGrid;
temperatureCombinedC(~isfinite(temperatureCombinedC)) = nan;

latitudeGrid = repmat(S.lat(:), 1, numel(S.lon));
petAnnualMm = opts.PETScale .* ...
    oudinAnnualPET(temperatureCombinedC, latitudeGrid);

secondsPerYear = 365.2425 .* 24 .* 60 .* 60;
precipitationAnnualRawMm = S.pGrid .* secondsPerYear;
positivePrecipitation = S.pGrid(isfinite(S.pGrid) & S.pGrid > 0);
if isempty(positivePrecipitation)
    error('OPI result contains no finite positive precipitation.');
end
precipitationThreshold = opts.MinimumPrecipitationFraction .* ...
    max(positivePrecipitation);
validMask = isfinite(S.pGrid) & S.pGrid > precipitationThreshold & ...
    isfinite(petAnnualMm) & petAnnualMm > 0 & ...
    isfinite(temperatureCombinedC);
if ~any(validMask, 'all')
    error('No valid cells remain after precipitation and PET masking.');
end

medianPrecipitation = median(S.pGrid(validMask));
medianPET = median(petAnnualMm(validMask));
precipitationRelative = S.pGrid ./ medianPrecipitation;
petRelative = petAnnualMm ./ medianPET;
aridityRelative = precipitationRelative ./ petRelative;
precipitationRelative(~validMask) = nan;
petRelative(~validMask) = nan;
aridityRelative(~validMask) = nan;

scales = opts.PrecipitationScales(:);
nScales = numel(scales);
gridSize = size(S.pGrid);
precipitationAnnualMm = nan([gridSize, nScales]);
aridityAbsolute = nan([gridSize, nScales]);
climaticWaterDeficitMm = nan([gridSize, nScales]);
for i = 1:nScales
    precipitationAnnualMm(:, :, i) = ...
        precipitationAnnualRawMm .* scales(i);
    aridityAbsolute(:, :, i) = ...
        precipitationAnnualMm(:, :, i) ./ petAnnualMm;
    climaticWaterDeficitMm(:, :, i) = ...
        precipitationAnnualMm(:, :, i) - petAnnualMm;
    thisAI = aridityAbsolute(:, :, i);
    thisCWD = climaticWaterDeficitMm(:, :, i);
    thisP = precipitationAnnualMm(:, :, i);
    thisAI(~validMask) = nan;
    thisCWD(~validMask) = nan;
    thisP(~validMask) = nan;
    aridityAbsolute(:, :, i) = thisAI;
    climaticWaterDeficitMm(:, :, i) = thisCWD;
    precipitationAnnualMm(:, :, i) = thisP;
end

summary = summarizeScenarios(scales, precipitationAnnualMm, ...
    petAnnualMm, aridityAbsolute, climaticWaterDeficitMm, ...
    validMask, latitudeGrid);

configuration = struct;
configuration.petMethod = "Oudin annual PET using constant annual-mean temperature";
configuration.petEquation = ...
    "PET=(Ra/2.45)*max(T+5,0)/100, integrated over midpoint months";
configuration.petScale = opts.PETScale;
configuration.precipitationScales = scales;
configuration.secondsPerYear = secondsPerYear;
configuration.minimumPrecipitationFraction = ...
    opts.MinimumPrecipitationFraction;
configuration.precipitationThresholdKgM2S = precipitationThreshold;
configuration.interpretation = ...
    "Relative aridity is primary; absolute P/PET values are scale scenarios.";

result = struct;
result.sourceResultFile = string(resultFile);
result.outputDir = string(outputDir);
result.lon = S.lon;
result.lat = S.lat;
result.hGridM = hGrid;
result.validMask = validMask;
result.temperatureState1C = temperatureState1C;
result.temperatureState2C = temperatureState2C;
result.temperatureCombinedC = temperatureCombinedC;
result.precipitationRawKgM2S = S.pGrid;
result.precipitationAnnualRawMm = precipitationAnnualRawMm;
result.precipitationRelative = precipitationRelative;
result.petAnnualMm = petAnnualMm;
result.petRelative = petRelative;
result.aridityRelative = aridityRelative;
result.precipitationAnnualMm = precipitationAnnualMm;
result.aridityAbsolute = aridityAbsolute;
result.climaticWaterDeficitMm = climaticWaterDeficitMm;
result.precipitationScales = scales;
result.summary = summary;
result.configuration = configuration;

if opts.WriteOutputs
    save(fullfile(outputDir, 'opiAridity_Results.mat'), 'result', '-v7.3');
    writetable(summary, fullfile(outputDir, ...
        'opiAridity_ScenarioSummary.csv'));
    writeReadme(fullfile(outputDir, 'README.txt'), resultFile, configuration);
end
if opts.MakeFigure
    makeAridityFigure(result, outputDir);
end

fprintf('Estimated OPI aridity from:\n%s\n', resultFile);
fprintf('Valid grid cells: %d of %d\n', ...
    sum(validMask, 'all'), numel(validMask));
fprintf('Output directory:\n%s\n', outputDir);
end

function hGrid = loadTopography(S, resultDir)
topoPath = fullfile(char(string(S.dataPath)), char(string(S.topoFile)));
if ~isfile(topoPath)
    topoPath = fullfile(resultDir, char(string(S.topoFile)));
end
if ~isfile(topoPath)
    error('Topography file referenced by OPI result was not found: %s', ...
        char(string(S.topoFile)));
end
topoVariables = who('-file', topoPath);
if ismember('hGrid', topoVariables)
    H = load(topoPath, 'hGrid');
    hGrid = H.hGrid;
else
    [~, ~, hGrid] = gridRead(topoPath);
end
hGrid(hGrid < 0) = 0;
window = tukeywin(size(hGrid, 1), S.rTukey) * ...
    tukeywin(size(hGrid, 2), S.rTukey)';
hGrid = window .* hGrid;
end

function petAnnualMm = oudinAnnualPET(temperatureC, latitudeDeg)
% Approximate annual Oudin PET using annual-mean T at all month midpoints.
monthMidpointDay = [15, 45, 74, 105, 135, 166, ...
    196, 227, 258, 288, 319, 349];
daysInMonth = [31, 28.2425, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
Gsc = 0.0820; % MJ m^-2 min^-1
latentHeat = 2.45; % MJ kg^-1; 1 kg m^-2 equals 1 mm water
phi = deg2rad(max(-89.9, min(89.9, latitudeDeg)));
temperatureFactor = max(temperatureC + 5, 0) ./ 100;
petAnnualMm = zeros(size(temperatureC));
for i = 1:numel(monthMidpointDay)
    J = monthMidpointDay(i);
    inverseDistance = 1 + 0.033 .* cos(2 .* pi .* J ./ 365);
    solarDeclination = 0.409 .* sin(2 .* pi .* J ./ 365 - 1.39);
    sunsetArgument = -tan(phi) .* tan(solarDeclination);
    sunsetArgument = max(-1, min(1, sunsetArgument));
    sunsetHourAngle = acos(sunsetArgument);
    ra = (24 .* 60 ./ pi) .* Gsc .* inverseDistance .* ...
        (sunsetHourAngle .* sin(phi) .* sin(solarDeclination) + ...
        cos(phi) .* cos(solarDeclination) .* sin(sunsetHourAngle));
    petDailyMm = (ra ./ latentHeat) .* temperatureFactor;
    petAnnualMm = petAnnualMm + petDailyMm .* daysInMonth(i);
end
petAnnualMm(~isfinite(temperatureC)) = nan;
end

function summary = summarizeScenarios(scales, precipitation, pet, ...
    aridity, cwd, validMask, latitudeGrid)
n = numel(scales);
medianPrecipitationMm = nan(n, 1);
medianPETMm = repmat(median(pet(validMask)), n, 1);
medianAridity = nan(n, 1);
medianCWDmm = nan(n, 1);
fractionArid = nan(n, 1);
fractionSemiarid = nan(n, 1);
fractionDrySubhumid = nan(n, 1);
fractionHumid = nan(n, 1);
areaWeight = cosd(latitudeGrid(validMask));
areaWeight = areaWeight ./ sum(areaWeight);
for i = 1:n
    thisP = precipitation(:, :, i);
    thisAI = aridity(:, :, i);
    thisCWD = cwd(:, :, i);
    values = thisAI(validMask);
    medianPrecipitationMm(i) = median(thisP(validMask));
    medianAridity(i) = median(values);
    medianCWDmm(i) = median(thisCWD(validMask));
    fractionArid(i) = sum(areaWeight(values < 0.20));
    fractionSemiarid(i) = sum(areaWeight(values >= 0.20 & values < 0.50));
    fractionDrySubhumid(i) = ...
        sum(areaWeight(values >= 0.50 & values < 0.65));
    fractionHumid(i) = sum(areaWeight(values >= 0.65));
end
summary = table(scales, medianPrecipitationMm, medianPETMm, ...
    medianAridity, medianCWDmm, fractionArid, fractionSemiarid, ...
    fractionDrySubhumid, fractionHumid, ...
    'VariableNames', {'precipitation_scale', ...
    'median_precipitation_mm_per_year', 'median_pet_mm_per_year', ...
    'median_p_over_pet', 'median_p_minus_pet_mm_per_year', ...
    'area_fraction_model_equivalent_arid', ...
    'area_fraction_model_equivalent_semiarid', ...
    'area_fraction_model_equivalent_dry_subhumid', ...
    'area_fraction_model_equivalent_humid'});
end

function makeAridityFigure(result, outputDir)
[~, referenceIndex] = min(abs(result.precipitationScales - 1));
absoluteAI = result.aridityAbsolute(:, :, referenceIndex);
fig = figure('Color', 'w', 'Name', 'OPI aridity estimate', ...
    'Position', [100, 100, 1450, 480]);
t = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

ax1 = nexttile(t);
imagesc(ax1, result.lon, result.lat, result.temperatureCombinedC);
set(ax1, 'YDir', 'normal'); axis(ax1, 'image'); colorbar(ax1);
xlabel(ax1, 'Longitude'); ylabel(ax1, 'Latitude');
title(ax1, 'Combined surface temperature (C)');

ax2 = nexttile(t);
imagesc(ax2, result.lon, result.lat, result.aridityRelative);
set(ax2, 'YDir', 'normal'); axis(ax2, 'image'); colorbar(ax2);
clim(ax2, robustColorLimits(result.aridityRelative));
xlabel(ax2, 'Longitude'); ylabel(ax2, 'Latitude');
title(ax2, 'Relative aridity index');

ax3 = nexttile(t);
imagesc(ax3, result.lon, result.lat, absoluteAI);
set(ax3, 'YDir', 'normal'); axis(ax3, 'image'); colorbar(ax3);
clim(ax3, robustColorLimits(absoluteAI));
xlabel(ax3, 'Longitude'); ylabel(ax3, 'Latitude');
title(ax3, sprintf('Scenario P/PET (P scale %.2g)', ...
    result.precipitationScales(referenceIndex)));

exportgraphics(fig, fullfile(outputDir, 'Fig_OPI_Aridity.png'), ...
    'Resolution', 220);
savefig(fig, fullfile(outputDir, 'Fig_OPI_Aridity.fig'));
close(fig);
end

function limits = robustColorLimits(values)
values = values(isfinite(values));
if isempty(values)
    limits = [0, 1];
    return
end
limits = prctile(values, [2, 98]);
if limits(1) == limits(2)
    padding = max(abs(limits(1)) .* 0.05, 1e-9);
    limits = limits + [-padding, padding];
end
end

function writeReadme(fileName, resultFile, configuration)
fid = fopen(fileName, 'w');
if fid == -1
    error('Could not create aridity README: %s', fileName);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'OPI aridity estimate\n');
fprintf(fid, 'Source: %s\n', resultFile);
fprintf(fid, 'PET method: %s\n', configuration.petMethod);
fprintf(fid, 'PET equation: %s\n', configuration.petEquation);
fprintf(fid, 'Precipitation scales: %s\n', ...
    strjoin(compose('%.4g', configuration.precipitationScales), ', '));
fprintf(fid, '\nInterpretation:\n');
fprintf(fid, ['aridityRelative is the primary within-case spatial index.\n' ...
    'aridityAbsolute and climaticWaterDeficitMm are scale scenarios.\n' ...
    'They are not calibrated annual paleoclimate reconstructions.\n']);
end
