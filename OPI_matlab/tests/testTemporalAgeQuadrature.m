function tests = testTemporalAgeQuadrature
tests = functiontests(localfunctions);
end

function testDirectionalWesterholdCooling(testCase)
config = makeConfig();
old = temporalAgeQuadrature(34, 34, 30, nan, nan, config);
young = temporalAgeQuadrature(20, 20, 30, nan, nan, config);

verifyGreaterThan(testCase, old.temperatureMeanShiftC, 0);
verifyLessThan(testCase, young.temperatureMeanShiftC, 0);
verifyEqual(testCase, old.temperatureMeanShiftC - ...
    young.temperatureMeanShiftC, 4.5, 'AbsTol', 1e-12);
end

function testIntervalContainingModelAgeIsStillMarginalized(testCase)
config = makeConfig();
temporal = temporalAgeQuadrature(29, 37, 30, 29, 37, config);

verifyEqual(testCase, numel(temporal.ageMa), 17);
verifyEqual(testCase, sum(temporal.weight), 1, 'AbsTol', 1e-12);
verifyGreaterThan(testCase, max(temporal.temperatureMeanShiftC) - ...
    min(temporal.temperatureMeanShiftC), 2);
verifyEqual(testCase, temporal.operatorDistanceMyr, zeros(1, 17), ...
    'AbsTol', 1e-12);
end

function testOperatorExtrapolationUsesEachAgeNode(testCase)
config = makeConfig();
temporal = temporalAgeQuadrature(24.8, 26.1, 30, 29, 37, config);

verifyGreaterThanOrEqual(testCase, min(temporal.operatorDistanceMyr), 2.9);
verifyLessThan(testCase, max(temporal.temperatureMeanShiftC), 0);
verifyGreaterThan(testCase, min(temporal.temperatureSigmaC), 0);
end

function config = makeConfig()
config = struct;
config.age_quadrature_nodes = 17;
config.model_time_slice_Ma = 30;
config.westerhold_old_age_Ma = 34;
config.westerhold_young_age_Ma = 20;
config.westerhold_temperature_change_C = 4.5;
config.westerhold_temperature_change_1sigma_C = 0.5;
config.westerhold_regional_response_mean = 1.0;
config.westerhold_regional_response_1sigma = 0.35;
config.temporal_temperature_rate_C_per_Myr = 0.15;
config.temporal_water_d18O_rate_permil_per_Myr = 0.25;
config.temporal_operator_extrapolation_rate_C_per_Myr = 0.5;
end
