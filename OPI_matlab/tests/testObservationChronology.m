function tests = testObservationChronology
tests = functiontests(localfunctions);
end

function testIntervalCompatibilityAt30Ma(testCase)
modelAgeMinMa = 28;
modelAgeMaxMa = 32;

[localStatus, localEligible, localDistance] = classifyObservationChronology( ...
    25.1, 33.7, "known", modelAgeMinMa, modelAgeMaxMa, "include_provisional");
[xiongStatus, xiongEligible, xiongDistance] = classifyObservationChronology( ...
    29, 37, "known", modelAgeMinMa, modelAgeMaxMa, "include_provisional");
[liStatus, liEligible, liDistance] = classifyObservationChronology( ...
    22, 26, "known", modelAgeMinMa, modelAgeMaxMa, "include_provisional");

verifyTrue(testCase, localEligible);
verifyTrue(testCase, xiongEligible);
verifyTrue(testCase, liEligible);
verifyEqual(testCase, localStatus, "age_interval_contains_model_time");
verifyEqual(testCase, xiongStatus, "age_interval_contains_model_time");
verifyEqual(testCase, liStatus, "age_offset_uncertainty_inflated");
verifyEqual(testCase, [localDistance, xiongDistance, liDistance], [0, 0, 4]);
end

function testAgeDistanceUsesNominalModelAge(testCase)
[status, eligible, distanceMyr] = classifyObservationChronology( ...
    26, 28, "known", 28, 32, "include_provisional");
verifyTrue(testCase, eligible);
verifyEqual(testCase, status, "age_offset_uncertainty_inflated");
verifyEqual(testCase, distanceMyr, 2);
end

function testUnknownAgeCanBeProvisionallyIncluded(testCase)
[status, eligible] = classifyObservationChronology( ...
    nan, nan, "unknown", 28, 32, "include_provisional");
verifyTrue(testCase, eligible);
verifyEqual(testCase, status, "age_unknown_provisional_include");
end

function testDistanceBetweenObservationAndOperatorIntervals(testCase)
verifyEqual(testCase, ageIntervalDistanceMyr(24.8, 26.1, 29, 37), 2.9, ...
    'AbsTol', 1e-12);
verifyEqual(testCase, ageIntervalDistanceMyr(29, 37, 29, 37), 0);
end
