function result = opiInferElevation_TerrazasWarmestML( ...
    lakeTemperatureC, sigmaLakeTemperatureC, latitude, ...
    T0WarmestC, sigmaT0C, lapseRateCPerKm, sigmaLapseRateCPerKm, varargin)
% opiInferElevation_TerrazasWarmestML infers elevation with explicit climate inputs.
%
% lakeTemperatureC must be the evaporation/mineral-offset-corrected warmest
% lake-water temperature. T0WarmestC is warmest-month sea-level air
% temperature, not the annual OPI T0 parameter unless that interpretation
% has been independently justified. Lapse rate is positive in deg C/km.

p = inputParser;
addParameter(p, 'LakeAreaKm2', []);
addParameter(p, 'LakeDepthM', []);
addParameter(p, 'ElevationGridKm', (0:0.01:6)');
addParameter(p, 'Model', []);
addParameter(p, 'ModelFile', "");
parse(p, varargin{:});
opts = p.Results;

validateattributes(lakeTemperatureC, {'numeric'}, {'vector', 'real'});
validateattributes(sigmaLakeTemperatureC, {'numeric'}, ...
    {'vector', 'real', 'positive'});
validateattributes(T0WarmestC, {'numeric'}, {'scalar', 'real', 'finite'});
validateattributes(sigmaT0C, {'numeric'}, {'scalar', 'real', 'nonnegative'});
validateattributes(lapseRateCPerKm, {'numeric'}, {'scalar', 'real', 'positive'});
validateattributes(sigmaLapseRateCPerKm, {'numeric'}, ...
    {'scalar', 'real', 'nonnegative'});

lakeTemperatureC = lakeTemperatureC(:);
n = numel(lakeTemperatureC);
sigmaLakeTemperatureC = expandVector(sigmaLakeTemperatureC, n, ...
    'sigmaLakeTemperatureC');
latitude = expandVector(latitude, n, 'latitude');
area = expandOptional(opts.LakeAreaKm2, n, 'LakeAreaKm2');
depth = expandOptional(opts.LakeDepthM, n, 'LakeDepthM');

good = isfinite(lakeTemperatureC) & isfinite(sigmaLakeTemperatureC) & ...
    sigmaLakeTemperatureC > 0 & isfinite(latitude);
if ~any(good)
    error('No finite lake-temperature observations are available.');
end
lakeTemperatureC = lakeTemperatureC(good);
sigmaLakeTemperatureC = sigmaLakeTemperatureC(good);
latitude = latitude(good);
if ~isempty(area), area = area(good); end
if ~isempty(depth), depth = depth(good); end

% Combine replicate proxy measurements before applying shared transfer-model
% uncertainty, which should not shrink as independent analytical noise.
weights = 1 ./ sigmaLakeTemperatureC.^2;
lakeTemperatureMeanC = sum(weights .* lakeTemperatureC) ./ sum(weights);
sigmaLakeTemperatureMeanC = sqrt(1 ./ sum(weights));
latitudeMean = sum(weights .* latitude) ./ sum(weights);
areaMean = finiteWeightedMean(area, weights);
depthMean = finiteWeightedMean(depth, weights);

z = opts.ElevationGridKm(:);
if isempty(z) || any(~isfinite(z)) || any(diff(z) <= 0) || any(z < 0)
    error('ElevationGridKm must be a finite, increasing, nonnegative vector.');
end
airC = T0WarmestC - lapseRateCPerKm .* z;
commonArgs = {'LakeAreaKm2', areaMean, 'LakeDepthM', depthMean, ...
    'Model', opts.Model, 'ModelFile', opts.ModelFile};
[lakePredC, sigmaTransferC, transferInfo] = ...
    lakeTransferML_TerrazasWarmest(airC, latitudeMean, ...
    "air_to_lake", commonArgs{:});

sigmaAirC = sqrt(sigmaT0C.^2 + (z .* sigmaLapseRateCPerKm).^2);
lakePlusC = lakeTransferML_TerrazasWarmest(airC + sigmaAirC, ...
    latitudeMean, "air_to_lake", commonArgs{:}, 'IncludeResidual', false);
lakeMinusC = lakeTransferML_TerrazasWarmest(airC - sigmaAirC, ...
    latitudeMean, "air_to_lake", commonArgs{:}, 'IncludeResidual', false);
sigmaClimatePropagatedC = abs(lakePlusC - lakeMinusC) ./ 2;
sigmaTotalC = sqrt(sigmaLakeTemperatureMeanC.^2 + ...
    sigmaTransferC.^2 + sigmaClimatePropagatedC.^2);

residualC = lakeTemperatureMeanC - lakePredC;
chi2 = (residualC ./ sigmaTotalC).^2;
[chi2Min, iBest] = min(chi2);
relativeLikelihood = exp(-0.5 .* (chi2 - chi2Min));
normalizer = trapz(z, relativeLikelihood);
if normalizer <= 0 || ~isfinite(normalizer)
    error('Elevation likelihood could not be normalized.');
end
posterior = relativeLikelihood ./ normalizer;
cdf = cumtrapz(z, posterior);
cdf = cdf ./ cdf(end);

result = struct;
result.bestElevationKm = z(iBest);
result.medianElevationKm = quantileFromCdf(z, cdf, 0.5);
result.ci68ElevationKm = [quantileFromCdf(z, cdf, 0.16), ...
    quantileFromCdf(z, cdf, 0.84)];
result.ci95ElevationKm = [quantileFromCdf(z, cdf, 0.025), ...
    quantileFromCdf(z, cdf, 0.975)];
result.minimumChi2 = chi2Min;
result.nObservations = numel(lakeTemperatureC);
result.lakeTemperatureMeanC = lakeTemperatureMeanC;
result.sigmaLakeTemperatureMeanC = sigmaLakeTemperatureMeanC;
result.latitude = latitudeMean;
result.T0WarmestC = T0WarmestC;
result.sigmaT0C = sigmaT0C;
result.lapseRateCPerKm = lapseRateCPerKm;
result.sigmaLapseRateCPerKm = sigmaLapseRateCPerKm;
result.transferInfo = transferInfo;
result.profile = table(z, airC, lakePredC, sigmaTransferC, ...
    sigmaClimatePropagatedC, sigmaTotalC, residualC, chi2, posterior, cdf, ...
    transferInfo.outsideGlobalTrainingRange, ...
    transferInfo.outsideHighElevationTrainingRange, ...
    'VariableNames', {'elevation_km', 'air_warmest_C', ...
    'predicted_lake_warmest_C', 'sigma_transfer_C', ...
    'sigma_T0_lapse_propagated_C', 'sigma_total_C', 'residual_C', ...
    'chi2', 'posterior_density', 'cumulative_probability', ...
    'outside_global_training_range', ...
    'outside_high_elevation_training_range'});
end

function value = expandVector(value, n, label)
if isscalar(value)
    value = repmat(double(value), n, 1);
elseif numel(value) == n
    value = double(value(:));
else
    error('%s must be scalar or have %d elements.', label, n);
end
end

function value = expandOptional(value, n, label)
if isempty(value)
    return
end
value = expandVector(value, n, label);
end

function value = finiteWeightedMean(x, weights)
if isempty(x)
    value = [];
    return
end
good = isfinite(x) & x > 0;
if ~any(good)
    value = [];
else
    value = sum(weights(good) .* x(good)) ./ sum(weights(good));
end
end

function value = quantileFromCdf(x, cdf, probability)
[cdfUnique, ia] = unique(cdf, 'stable');
xUnique = x(ia);
if numel(cdfUnique) < 2
    value = x(1);
else
    value = interp1(cdfUnique, xUnique, probability, 'linear', 'extrap');
    value = min(max(value, x(1)), x(end));
end
end
