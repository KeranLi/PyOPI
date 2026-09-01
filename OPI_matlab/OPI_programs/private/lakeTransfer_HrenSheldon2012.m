function [TOut, sigmaT, model] = lakeTransfer_HrenSheldon2012(TIn, season, direction)
% lakeTransfer_HrenSheldon2012 applies Hren and Sheldon (2012) lake
% surface-water to mean-annual-air temperature transfer functions.
%
% TIn and TOut are in degrees C. The published equations are in the
% water_to_air direction: MAAT = a*Tw^2 + b*Tw + c. The air_to_water
% direction solves the quadratic and returns the lower plausible lake
% surface-water temperature root. The warmest-month equation is refit from
% the article appendix for northern hemisphere, non-tropical lakes.

if nargin < 2 || isempty(season)
    season = "annual";
end
if nargin < 3 || isempty(direction)
    direction = "water_to_air";
end

season = lower(string(season));
direction = lower(string(direction));
[a, b, c, sigmaResidual, r2, seasonLabel] = coefficients(season);

TIn = TIn(:);
switch direction
    case {"water_to_air", "lake_to_air", "tw_to_maat"}
        TOut = a.*TIn.^2 + b.*TIn + c;
    case {"air_to_water", "air_to_lake", "maat_to_tw"}
        TOut = invertQuadratic(TIn, a, b, c);
    otherwise
        error('Unknown Hren-Sheldon transfer direction: %s', direction);
end

sigmaT = repmat(sigmaResidual, size(TOut));

model = struct;
model.name = "HrenSheldon2012_" + seasonLabel + "_" + direction;
model.reference = "Hren and Sheldon, 2012, EPSL 337-338, 77-84";
model.season = seasonLabel;
model.direction = direction;
model.equation = "MAAT_C = a*Tw_C^2 + b*Tw_C + c";
model.a = a;
model.b = b;
model.c = c;
model.r2 = r2;
model.sigmaResidual_C = sigmaResidual;
end

function Tw = invertQuadratic(MAAT, a, b, c)
Tw = nan(size(MAAT));
for i = 1:numel(MAAT)
    if ~isfinite(MAAT(i))
        continue
    end
    rootsTw = roots([a, b, c - MAAT(i)]);
    rootsTw = real(rootsTw(abs(imag(rootsTw)) < 1e-8));
    rootsTw = rootsTw(isfinite(rootsTw));
    plausible = rootsTw(rootsTw >= -20 & rootsTw <= 45);
    if isempty(plausible)
        plausible = rootsTw;
    end
    if isempty(plausible)
        continue
    end
    % The high root is usually the nonphysical branch of the parabola.
    Tw(i) = min(plausible);
end
end

function [a, b, c, sigmaResidual, r2, seasonLabel] = coefficients(season)
switch season
    case {"annual", "ann", "malst"}
        a = -0.0318;
        b = 2.195;
        c = -12.607;
        sigmaResidual = 1.62;
        r2 = 0.96;
        seasonLabel = "annual";
    case {"amj", "spring"}
        a = -0.0097;
        b = 1.379;
        c = -8.23;
        sigmaResidual = 2.12;
        r2 = 0.935;
        seasonLabel = "amj";
    case {"jja", "summer"}
        a = -0.0055;
        b = 1.476;
        c = -18.915;
        sigmaResidual = 2.69;
        r2 = 0.90;
        seasonLabel = "jja";
    case {"amjjaso", "ao", "spring_fall", "apr_oct", "growing"}
        a = -0.0146;
        b = 1.753;
        c = -16.079;
        sigmaResidual = 1.89;
        r2 = 0.949;
        seasonLabel = "amjjaso";
    case {"warmest", "warmest_month", "wmmwt"}
        a = -0.003122;
        b = 1.501709;
        c = -22.654929;
        sigmaResidual = 3.02;
        r2 = 0.8696;
        seasonLabel = "warmest";
    otherwise
        error('Unknown Hren-Sheldon season: %s', season);
end
end
