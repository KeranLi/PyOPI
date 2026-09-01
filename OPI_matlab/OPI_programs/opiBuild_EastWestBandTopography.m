function hGrid = opiBuild_EastWestBandTopography( ...
    hGrid, lat, gangdese, qiangtang, valleyMode, valleyElevation)
% Impose Gangdese, Qiangtang, and valley geometry using latitude-only bands.

hGrid = imposeBand(hGrid, lat, 29.15, 0.35, 0.45, gangdese);
hGrid = imposeBand(hGrid, lat, 33.20, 0.45, 0.45, qiangtang);
if string(valleyMode) == "Vnone"
    hGrid = imposeRamp(hGrid, lat, gangdese, qiangtang);
else
    if ~isfinite(valleyElevation)
        error('A finite valley elevation is required for %s.', valleyMode);
    end
    hGrid = imposeBand(hGrid, lat, 31.30, 0.45, 0.55, valleyElevation);
end
hGrid = max(hGrid, 0);
end

function hGrid = imposeBand(hGrid, lat, center, halfWidth, transition, target)
w = bandWeight(lat, center, halfWidth, transition);
for i = 1:size(hGrid, 1)
    if w(i) > 0
        hGrid(i, :) = (1 - w(i)) .* hGrid(i, :) + w(i) .* target;
    end
end
end

function hGrid = imposeRamp(hGrid, lat, southTarget, northTarget)
center = 31.30;
halfWidth = 0.45;
transition = 0.55;
w = bandWeight(lat, center, halfWidth, transition);
south = center - halfWidth - transition;
north = center + halfWidth + transition;
for i = 1:size(hGrid, 1)
    if w(i) > 0
        f = min(max((lat(i) - south) / (north - south), 0), 1);
        target = southTarget + f * (northTarget - southTarget);
        hGrid(i, :) = (1 - w(i)) .* hGrid(i, :) + w(i) .* target;
    end
end
end

function w = bandWeight(lat, center, halfWidth, transition)
d = abs(lat(:) - center);
w = zeros(size(d));
w(d <= halfWidth) = 1;
mask = d > halfWidth & d < halfWidth + transition;
x = (d(mask) - halfWidth) ./ transition;
w(mask) = 1 - x .* x .* (3 - 2 .* x);
end
