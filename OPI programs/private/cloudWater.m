function [zRhoC, rhoC, zCMean, z248Path, z268Path] = ...
    cloudWater(xPath, yPath, zMax, ...
    x, y, hGrid, U, azimuth, NM, fC, kappa, tauC, tauF, hRho, ...
    zBar, T, gammaEnv, gammaSat, gammaRatio, rhoS0, hV)
% Calculates cloudwater density for specified section.
% Input arguments
% xPath, yPath: eastings and northings for section path (vectors, m).
% zMax: sea level elevation for top of rhoC grid (scalar, m).
% x, y = grid vectors indicating coordinates for geographic grids
%   (e.g. hGrid), with x and y oriented in the east and north directions,
%   respectively (vectors with lengths nX and nY, m).
% hGrid = geographic grid for topography, with x (east) in the row
%   direction, and y (north) in the column direction (matrix, nY x nX, m).
% U = base-state wind speed (scalar, m/s).
% azimuth = base-state wind direction (down wind) (scalar, degrees).
% NM = saturation buoyancy frequency (scalar, rad/s).
% fC = Coriolis frequency (scalar, rad/s).
% hRho = scale height for density (scalar, m).
% fVGrid: vapor-ratio field [dimensionless] (matrix)
% gammaMoist, gammaEnv: moist and environmental lapse rates [K/m] (scalars).
% rhoS0: saturated water vapor density at sea level [kg/m^3] (scalar).
% hV: mean height of water vapor [m] (scalar).
% tauC: mean residence-time for cloud water [s] (scalar).
% kappa: horizontal eddy diffusivity [m^2/s] (scalar).
% Output arguments
% hRhoC: grid vector with sea level elevations for rhoC grid [m] (vector).
% rhoC: grid indicating cloudwater density [kg/m^3] (matrix).
% zCMean: Mean height above ground surface of cloud water [m] (scalar).
%
% Aug 20, 2019: 
% Singularities caused by abs(U*kS) = abs(fC) are treated by:
%   hHat((abs(denominator)<(U*dkS/2)^2) = 0, 
% where fC is the Coriolis frequency.

% Mark Brandon, Yale University, 2020

%% Initialize variables
%... Vertical dimension for cloud-water density, for z = 0 to zMax
% Recommended: nZ = 100
nZ = 100;

%% %% Compute reference solution
%... Get Fourier solution for Euler equations
 [s, t, ~, ~, ~, kS, kT, hHat, kZ] = ...
    fourierSolution(x, y, hGrid, U, azimuth, NM, fC, hRho);

%... Parameters for wind grids
dS = s(2)-s(1);
nS = length(s);
nT = length(t);

%... Calculate PStarHat, reference precipitation rate for wave domain
GSHat = gammaRatio*rhoS0*1i*kS.*U./(1 - 1i*kZ*hV);
GCHat = 1./(tauC*(kappa*(kS.^2 + kT.^2) - 1i*kS*U) + 1);
GFHat = 1./(tauF*(kappa*(kS.^2 + kT.^2) - 1i*kS*U) + 1);
pStarHat = GSHat.*GCHat.*GFHat.*hHat;

%... Transform back to space domain, remove padding, and 
% set negative values in pStarPosWind to zero. The 'symmetric" option
% for ifft2 indicates that PStarHat is conjugate symmetric, which ensures
% that pStarWind is returned as a real-valued grid.
pStarPosWind = ifft2(pStarHat, 'symmetric');
clear pStarHat
pStarPosWind = pStarPosWind(1:nS, 1:nT);
pStarPosWind(pStarPosWind<0) = 0;

%% Calculate vapor ratio, fV
%... Calculate column-density fields for cloud water QC, 
% falling precipitation QF, and the total moisture QT. These
% fields are for the reference solution, and are truncated  to 
% postive values, as required for the moisture-balance calculation.
QCStarPosWind = ifft2(tauC*GSHat.*GCHat.*hHat, 'symmetric');
QCStarPosWind = QCStarPosWind(1:nS, 1:nT);
QCStarPosWind(QCStarPosWind<0) = 0;

QFStarPosWind = ifft2(tauF*GSHat.*GCHat.*GFHat.*hHat, 'symmetric');
QFStarPosWind = QFStarPosWind(1:nS, 1:nT);
QFStarPosWind(QFStarPosWind<0) = 0;
clear GSHat GCHat GFHat

QTStarPosWind = rhoS0*hV + QCStarPosWind + QFStarPosWind;
clear QCStarPosWind QFStarPosWind

%... Integrate along the columns of the s,t grid (+s direction) to
% calculate the water-vapor ratio, fV.
fVWind = (rhoS0*hV./QTStarPosWind) ...
    .*exp(-cumtrapz(pStarPosWind.*dS./(U*QTStarPosWind))); 
clear pStarPosWind

%% Compute cloud-water density in section along path
%... Start with reference cloud-water density field at zBar = 0
% Convolution form: rhoCStarHat0 = tauC*gSHat0.*gCHat0.*hHat;
% gSHat0 = gammaRatio*rhoS0.*1i.*kS*U/hV;
% gCHat0 = 1/(tauC*(kappa*(kS.^2 + kT.^2) + 1i*kS*U) + 1);
% rhoCStarHat = rhoCStarHat0.*exp((1i.*kZ + 1/(2*hRho) - 1/hV)*zBar)
rhoCStarHat0 = (gammaRatio*rhoS0.*1i.*kS*U/hV).*hHat ...
    ./(kappa*(kS.^2 + kT.^2) + 1i*kS*U + 1/tauC);

%... Calculate cloud-water density in specified section
% Initialize variables
zBarRhoC = linspace(0, zMax, nZ)';
nPath = length(xPath);
zRhoC = zeros(nZ, nPath);
rhoC = zeros(nZ, nPath);
% Calculate s,t coordinates for path (maintains right-handed s,t,z)
sPath = xPath*sind(azimuth) + yPath*cosd(azimuth);
tPath = -xPath*cosd(azimuth) + yPath*sind(azimuth);
% Calculate rhoC at increments of zBar
for i = 1:nZ
    % Calculate rhoC for current zBar = zBarRhoC(i)
    rhoCForZBar = ...
        ifft2(rhoCStarHat0.*exp((1i.*kZ + 1/(2*hRho)- 1/hV).*zBarRhoC(i)), ...
        'symmetric');
    rhoCForZBar = fVWind.*rhoCForZBar(1:nS, 1:nT);
    F = griddedInterpolant({s, t}, rhoCForZBar, 'linear', 'linear');    
    rhoC(i,:) = F(sPath, tPath);
    % Actually elevation, z, for path points at zBar = zBarRhoC(i)
    zForZBar = zBarRhoC(i) ...
        + ifft2(hHat.*exp((1i.*kZ - 1/(2*hRho)).*zBarRhoC(i)), 'symmetric');
    zForZBar = zForZBar(1:nS, 1:nT);
    F = griddedInterpolant({s, t}, zForZBar, 'linear', 'linear');    
    zRhoC(i,:) = F(sPath, tPath);
end
%... rhoC is now in terms of zBar (height above base of model)
% The positive part of this variable is saved here as rhoCPosForZbar
% to be used below to calculate the weighted mean height of cloud water.
rhoCPosForZBar = rhoC;
%... Interpolate rhoC results to a regular grid, with the
% same vertical grid spacing as zBarRhoC.
for j = 1:nPath
    rhoC(:,j) = interp1(zRhoC(:,j), rhoC(:,j), zBarRhoC, ...
        'linear', 'extrap');
end
zRhoC = zBarRhoC;

% Weighted mean height of cloud water relative to the land surface
rhoCPosForZBar(rhoCPosForZBar<=0) = nan;
zCMean = sum(zBarRhoC.*rhoCPosForZBar,'all', 'omitnan') ...
    ./sum(rhoCPosForZBar,'all','omitnan');
clear zBarRhoC rhoCPosForZBar

%... Calculate height along path of the 268 K and 248 K isotherms
z248 = isotherm(248, zBar, T, gammaEnv, gammaSat, hRho, nS, nT, hHat, kZ);
z248Path = interp2(t, s, z248, tPath, sPath);    
z268 = isotherm(268, zBar, T, gammaEnv, gammaSat, hRho, nS, nT, hHat, kZ);
z268Path = interp2(t, s, z268, tPath, sPath);    

end
