function uPrimeGrid = uPrime( ...
    zBar, x, y, hGrid, U, azimuth, NM, fC, hRho, isFirst)
% Calculate horizontal perturbation velocity, uPrime, oriented in
% the wind direction.
% Input arguments
% zBar: base-state height for velocity ratio grid (scalar m).
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
% uPrimeGrid: grid for uPrime at zBar (matrix, nY x nX, m).

% Mark Brandon, Yale University, 2021

%% Initialize variables
persistent s t nS nT kS kT kZ hHat

%% Compute Fourier solution (if needed)
if isFirst==true
    [s, t, Sxy, Txy, ~, kS, kT, hHat, kZ] = ...
        fourierSolution(x, y, hGrid, U, azimuth, NM, fC, hRho);
    %... Parameters for wind grids
    nS = length(s);
    nT = length(t);
end

%% Compute uPrimeGrid
%... Calculate horizontal perturbation velocity, uPrime.
% Note that equation has a singularity when kS==0, but terms
% with kS==0 can be set to zero given that vertical velocity
% % is zero when kS==0.
uPrimeHat = (1i.*kS + fC.*kT./(U.*kS))./(kS.^2 + kT.^2) ...
    .*(1i*kZ + 1/(2*hRho)) ...
     .*1i.*kS.*U.*hHat.*exp((1i*kZ + 1/(2*hRho))*zBar);
uPrimeHat(1,1:end) = 0;
%... Transform to space domain, and remove padding.
% uPrimeGrid starts as a wind grid, ends as a geographic grid.
uPrimeGrid = ifft2(uPrimeHat, 'symmetric');
%clear uPrimeHat
uPrimeGrid = uPrimeGrid(1:nS, 1:nT);
%... Transform to geographic coordinates
F = griddedInterpolant({s, t}, uPrimeGrid, 'linear', 'none');
uPrimeGrid = F(Sxy, Txy);

end

