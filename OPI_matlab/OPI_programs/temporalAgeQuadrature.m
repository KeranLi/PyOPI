function temporal = temporalAgeQuadrature( ...
    ageMinMa, ageMaxMa, temperatureReferenceAgeMa, ...
    operatorAgeMinMa, operatorAgeMaxMa, config)
% Build an interval-age quadrature with a directional climate correction.


if ~all(isfinite([ageMinMa, ageMaxMa, temperatureReferenceAgeMa])) || ...
        ageMinMa > ageMaxMa
    error('Age quadrature requires finite ordered ages and a reference age.');
end
n = round(config.age_quadrature_nodes);
if ~isfinite(n) || n < 1
    error('age_quadrature_nodes must be a positive integer.');
end
if ageMinMa == ageMaxMa
    ageMa = ageMinMa;
    weight = 1;
else
    edges = linspace(ageMinMa, ageMaxMa, n + 1);
    ageMa = 0.5 .* (edges(1:end-1) + edges(2:end));
    weight = repmat(1 ./ n, 1, n);
end

durationMyr = config.westerhold_old_age_Ma - ...
    config.westerhold_young_age_Ma;
if ~isfinite(durationMyr) || durationMyr <= 0
    error('Westerhold old and young ages must define a positive duration.');
end
slopeCPerMyr = config.westerhold_temperature_change_C ./ durationMyr;
slopeSigmaCPerMyr = ...
    config.westerhold_temperature_change_1sigma_C ./ durationMyr;
ageOffsetMyr = ageMa - temperatureReferenceAgeMa;
temperatureMeanShiftC = config.westerhold_regional_response_mean .* ...
    slopeCPerMyr .* ageOffsetMyr;
temperatureTrendSigmaC = hypot( ...
    abs(config.westerhold_regional_response_mean .* ageOffsetMyr) .* ...
    slopeSigmaCPerMyr, ...
    abs(slopeCPerMyr .* ageOffsetMyr) .* ...
    config.westerhold_regional_response_1sigma);
temperatureResidualSigmaC = ...
    config.temporal_temperature_rate_C_per_Myr .* abs(ageOffsetMyr);

operatorDistanceMyr = zeros(size(ageMa));
if all(isfinite([operatorAgeMinMa, operatorAgeMaxMa]))
    if operatorAgeMinMa > operatorAgeMaxMa
        error('Operator calibration ages must be ordered.');
    end
    operatorDistanceMyr = max([operatorAgeMinMa - ageMa; ...
        ageMa - operatorAgeMaxMa; zeros(size(ageMa))], [], 1);
end
operatorSigmaC = config.temporal_operator_extrapolation_rate_C_per_Myr .* ...
    operatorDistanceMyr;
temperatureSigmaC = hypot(hypot(temperatureTrendSigmaC, ...
    temperatureResidualSigmaC), operatorSigmaC);

waterAgeOffsetMyr = ageMa - config.model_time_slice_Ma;
waterD18OSigmaPermil = ...
    config.temporal_water_d18O_rate_permil_per_Myr .* ...
    abs(waterAgeOffsetMyr);

temporal = struct;
temporal.ageMa = ageMa;
temporal.weight = weight;
temporal.temperatureReferenceAgeMa = temperatureReferenceAgeMa;
temporal.temperatureMeanShiftC = temperatureMeanShiftC;
temporal.temperatureSigmaC = temperatureSigmaC;
temporal.waterD18OSigmaPermil = waterD18OSigmaPermil;
temporal.operatorDistanceMyr = operatorDistanceMyr;
temporal.westerholdSlopeCPerMyr = slopeCPerMyr;
end
