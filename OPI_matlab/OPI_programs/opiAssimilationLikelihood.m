function [logLikelihoodJoint, detail] = opiAssimilationLikelihood( ...
    d18OPredPermil, airWarmestC, latitude, observation, varargin)
% opiAssimilationLikelihood evaluates the shared OPI observation operator.
%
% This function is the bridge for a future continuous optimizer: OPI supplies
% precipitation d18O and an explicitly seasonal air temperature, while the
% Terrazas forward model maps air temperature to the observed warmest lake T.

p = inputParser;
addParameter(p, 'LakeAreaKm2', []);
addParameter(p, 'LakeDepthM', []);
addParameter(p, 'D18OModelDiscrepancyPermil', 0, ...
    @(x) isscalar(x) && isfinite(x) && x >= 0);
addParameter(p, 'TemperatureModelDiscrepancyC', 0, ...
    @(x) isscalar(x) && isfinite(x) && x >= 0);
addParameter(p, 'Model', []);
addParameter(p, 'ModelFile', "");
parse(p, varargin{:});
opts = p.Results;

required = ["target_d18O_permil", "sigma_target_d18O_permil", ...
    "lake_temperature_warmest_C", "sigma_analytical_mean_C"];
missing = required(~isfield(observation, required));
if ~isempty(missing)
    error('Assimilation observation missing field(s): %s', ...
        strjoin(missing, ', '));
end

n = max([numel(d18OPredPermil), numel(airWarmestC), numel(latitude), 1]);
d18OPredPermil = expandInput(d18OPredPermil, n, 'd18OPredPermil');
airWarmestC = expandInput(airWarmestC, n, 'airWarmestC');
latitude = expandInput(latitude, n, 'latitude');

[lakePredC, sigmaTransferC, transferInfo] = ...
    lakeTransferML_TerrazasWarmest(airWarmestC, latitude, ...
    "air_to_lake", 'LakeAreaKm2', opts.LakeAreaKm2, ...
    'LakeDepthM', opts.LakeDepthM, 'Model', opts.Model, ...
    'ModelFile', opts.ModelFile);

sigmaTemperatureC = sqrt(observation.sigma_analytical_mean_C.^2 + ...
    sigmaTransferC.^2 + opts.TemperatureModelDiscrepancyC.^2);
temperatureResidualC = observation.lake_temperature_warmest_C - lakePredC;
logLikelihoodTemperature = normalLogLikelihood( ...
    temperatureResidualC, sigmaTemperatureC);

sigmaD18OPermil = sqrt(observation.sigma_target_d18O_permil.^2 + ...
    opts.D18OModelDiscrepancyPermil.^2);
d18OResidualPermil = observation.target_d18O_permil - d18OPredPermil;
logLikelihoodD18O = normalLogLikelihood(d18OResidualPermil, sigmaD18OPermil);
logLikelihoodJoint = logLikelihoodD18O + logLikelihoodTemperature;

detail = struct;
detail.logLikelihoodD18O = logLikelihoodD18O;
detail.logLikelihoodTemperature = logLikelihoodTemperature;
detail.d18OResidualPermil = d18OResidualPermil;
detail.sigmaD18OPermil = repmat(sigmaD18OPermil, n, 1);
detail.lakePredictedC = lakePredC;
detail.temperatureResidualC = temperatureResidualC;
detail.sigmaTemperatureC = sigmaTemperatureC;
detail.transferInfo = transferInfo;
end

function value = expandInput(value, n, label)
if isscalar(value)
    value = repmat(double(value), n, 1);
elseif numel(value) == n
    value = double(value(:));
else
    error('%s must be scalar or have %d elements.', label, n);
end
end

function logLikelihood = normalLogLikelihood(residual, sigma)
logLikelihood = -0.5 .* (residual ./ sigma).^2 - log(sigma) ...
    - 0.5 .* log(2*pi);
end
