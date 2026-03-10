function [xL, yL, zL, sL] = streamline( ...
    xL0, yL0, zL0, x, y, hGrid, U, azimuth, NM, fC, hRho, isFirst)
% Calculate single streamline starting from specified base-state point.
% Input arguments
% xL0, yL0, zL0: base-state coordinates for start of streamline 
%   (scalars m).
% x, y: grid vectors indicating coordinates for geographic grids
%   (e.g. hGrid), with x and y oriented in the east and north directions,
%   respectively (vectors with lengths nX and nY, m).
% hGrid: geographic grid for topography, with x (east) in the row
%   direction, and y (north) in the column direction (matrix, nY x nX, m).
% NM: saturation buoyancy frequency (scalar, rad/s).
% fC: Coriolis frequency (scalar, rad/s).
% hRho: scale height for density (scalar, m).
% isFirst: Set to true on first call, and to false on subsequent calls,
%   for a sequence of calls where the parameters, x, y, hGrid, U,
%   azimuth, NM, fC, and hRho remain unchanged.
% Output arguments
% xL, yL, zL: x, y, z coordinates for streamline ordered by increasing
%   time (vectors, m).
% sL: horizontal distance along streamline path from start (vector, m).

% Mark Brandon, Yale University, 2016-2020

%% Initialize variables
persistent zS0Prev sBar tBar nS nT kS kT kZ hHat sSWind tSWind zSWind 

%% Compute Fourier solution (if needed)
if isFirst==true
    zS0Prev = nan;
    [sBar, tBar, ~, ~, ~, kS, kT, hHat, kZ] = ...
    fourierSolution(x, y, hGrid, U, azimuth, NM, fC, hRho);
    %... Parameters for wind grids
    nS = length(sBar);
    nT = length(tBar);
end

%% Calculate stream surface in wind coordinates at zBar = zS0
if isFirst==true || zL0~=zS0Prev
    zS0Prev = zL0;
    %... Calculate stream surface
    % The Fourier integration operation requires that DC term for the
    % Fourier coefficients is zero (i.e. the "signal" has zero mean). 
    % This issue does not effect the integration of wPrime, because 
    % this operation simply reverses the original differentiation used 
    % to get wPrime from h. It is an issue for integration of uPrime and
    % vPrime, but we know that these variables are approximately zero 
    % mean over their range. The easiest way to adjust the result for 
    % this constraint is to set xPrimeHat(1, :) = 0 and 
    % yPrimeHat(1, :) = 0, as done below.
    zPrimeHat = hHat.*exp((1i*kZ + 1/(2*hRho))*zL0);
    sPrimeHat = (1i*kS + fC.*kT./(U*kS))./(kS.^2 + kT.^2) ...
        .*(1i*kZ + 1/(2*hRho)).*zPrimeHat;
    sPrimeHat(1,1:end) = 0;
    tPrimeHat = (1i*kT - fC/U)./(kS.^2 + kT.^2) ...
        .*(1i*kZ + 1/(2*hRho)).*zPrimeHat;
    tPrimeHat(1,1:end) = 0;
    %... Transform to space domain, and remove padding
    zSWind = zL0 + ifft2(zPrimeHat, 'symmetric'); 
    %clear zPrimeHat
    zSWind = zSWind(1:nS, 1:nT);
    sSWind = ifft2(sPrimeHat, 'symmetric');
    %clear sPrimeHat
    sSWind = sBar + sSWind(1:nS, 1:nT);
    tSWind = ifft2(tPrimeHat, 'symmetric');
    %clear tPrimeHat
    tSWind = tBar + tSWind(1:nS, 1:nT);
end

%% Calculate streamlines
%... Calculate xBar, yBar coordinates along path
[xBarPath, yBarPath] = windPath(xL0, yL0, azimuth, x, y);
%... Calculate sBar,tBar coordinates for path (s,t,z is right handed)
sBarPath = xBarPath*sind(azimuth) + yBarPath*cosd(azimuth);
tBarPath = -xBarPath*cosd(azimuth) + yBarPath*sind(azimuth);
%... Interpolate to get xL, yL, and zL along path
sL = interp2(tBar, sBar, sSWind, tBarPath, sBarPath);
tL = interp2(tBar, sBar, tSWind, tBarPath, sBarPath);
xL = sL*sind(azimuth) - tL*cosd(azimuth);
yL = sL*cosd(azimuth) + tL*sind(azimuth);
zL = interp2(tBar, sBar, zSWind, tBarPath, sBarPath);

end

