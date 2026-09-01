function distanceMyr = ageIntervalDistanceMyr( ...
    ageMinMa, ageMaxMa, referenceMinMa, referenceMaxMa)
% Minimum distance between two closed age intervals.


values = [ageMinMa, ageMaxMa, referenceMinMa, referenceMaxMa];
if ~all(isfinite(values)) || ageMinMa > ageMaxMa || ...
        referenceMinMa > referenceMaxMa
    error('Age intervals must have finite ordered bounds.');
end
distanceMyr = max([referenceMinMa - ageMaxMa, ...
    ageMinMa - referenceMaxMa, 0]);
end
