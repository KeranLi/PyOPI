function [temporalStatus, isEligible, ageDistanceMyr] = ...
    classifyObservationChronology( ...
    ageMinMa, ageMaxMa, chronologyStatus, modelAgeMinMa, ...
    modelAgeMaxMa, unknownAgeAction)
% Classify an interval-censored observation against one model time slice.


chronologyStatus = lower(string(chronologyStatus));
if chronologyStatus == "known"
    if ~all(isfinite([ageMinMa, ageMaxMa])) || ageMinMa > ageMaxMa
        error('Known chronology requires a valid finite age interval.');
    end
    modelTimeMa = 0.5 .* (modelAgeMinMa + modelAgeMaxMa);
    ageDistanceMyr = max([ageMinMa - modelTimeMa, ...
        modelTimeMa - ageMaxMa, 0]);
    isEligible = true;
    if ageDistanceMyr == 0
        temporalStatus = "age_interval_contains_model_time";
    else
        temporalStatus = "age_offset_uncertainty_inflated";
    end
else
    temporalStatus = "age_unknown_provisional_include";
    isEligible = lower(string(unknownAgeAction)) == "include_provisional";
    ageDistanceMyr = nan;
end
end
