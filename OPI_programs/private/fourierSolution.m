function [s, t, Sxy, Txy, hWind, kS, kT, hHat, kZ] = ...
    fourierSolution(x, y, hGrid, U, azimuth, NM, fC, hRho)
% Calculates the Fourier solution for the linearized Euler equations
% for flow of air over topography. The solution is based on saturated
% base state with a uniform buoyancy frequency, NM, and also uses the
% anelastic approximation, which allows for vertical variation in density
% in the base state. The solution comes from Durran and Klemp, 1982.
% Input arguments:
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
% Output arguments:
% s, t = grid vectors indicating coordinates for wind grids (e.g., hWind),
%   with +s oriented in the down-wind direction, and +t oriented 
%   90 degrees counterclockwise (s,t,z is right handed).
%   (vectors with lengths nS and nt, m).
% Sxy, Txy = grids containing the s and t coordinates, respectively, for
%   nodes in a geographic grid. These grids provide the basis for 
%   interpolating field variables from the wind-grid solution, back onto 
%   grid nodes associated with the original geographic grid
%    (matrices, nY x nX, m).
% hWind = wind grid for topography (matrix, nY x nX, m).
% kS, kT = grid vectors indicating wavenumber coordinates for the
%   Fourier coefficients in hHat, which has a wind-grid orientation
%   (vectors with lengths nS x nT, rad/m).
% hHat = grid with Fourier coefficients for topography (matrix, 
%   complex, nSPad x nTPad, m).
% kZ = grid of vertical wavenumbers for the Fourier solution of the
%   Euler equations, in wind-grid orientation
%   (matrix, complex, nSPad x nTPad, rad/m).
%
% Mark Brandon, Yale University, 2016-2020

%% Initialize system
warning('off', 'MATLAB:griddedInterpolant:MeshgridEval2DWarnId');

%% Compute
%... Transform topography to wind coordinates (s,t,z is right handed)
[Sxy, Txy, s, t, Xst, Yst] = windGrid(x, y, azimuth);
% Note: griddedInterpolant uses the ndgrid format, so that the order
% of the grid vectors, x and y, are reversed to account for the
% meshgrid format for hGrid. Also note that grid vectors must be
% specified as cell variables. The setting 'none' causes extrapolated
% values to be set to nans, which are then found and set to zero.
F = griddedInterpolant({y, x}, hGrid, 'linear', 'none');
hWind = F(Yst, Xst);
hWind(isnan(hWind)) = 0;
clear Xst Yst

%... Parameters for wind grid for topography
dS = s(2)-s(1);
dT = t(2)-t(1);
[nS, nT] = size(hWind);

%... Add zero padding around topography, to account for the full
% response caused by lifting over the topography. For example, an
% increase NM or U, or a decrease tauC or tauF will cause lifting
% and precipitatin to shift upwind, and vice versa. If the grid is
% too small to contain these responses, then the features will appear
% on the other size of the grid (a phenomenon called "wraparound").
% In my experience, zero padding in the wind direction is helpful.
% The recommendation is nSPad = 2*nS, nTPad = nT.
nSPad = 2*nS;
nTPad = nT;

%... Calculate wavenumber grids for topography (rad/m)
% Wavenumbers for s direction (wind direction)
i_kSMostNeg = ceil(nSPad/2)+1;
kS = (0:nSPad-1)'/nSPad;
kS(i_kSMostNeg:nSPad) = kS(i_kSMostNeg:nSPad)-1;
kS = 2*pi*kS/dS;
dKS = kS(2) - kS(1);
% Wavenumbers for t direction
i_kTMostNeg = ceil(nTPad/2)+1;
kT = (0:nTPad-1)/nTPad;
kT(i_kTMostNeg:nTPad) = kT(i_kTMostNeg:nTPad)-1;
kT = 2*pi*kT/dT;

%... Calculate fourier transform of topography
hHat = fft2(hWind, nSPad, nTPad);

%... Calculate denominator for kZ equation
% This step is used to avoid singularies where abs(U*kS)==abs(fC).
% My method, from Queney, 1947, p. 46-48, uses these corrections:
% hHat((abs(denominator)<(U*dkS/2)^2) = 0, and
% demoninator(denominator==0) = eps. The justification is that, at
% this singularity, the vertical velocities go to zero.
% The modification to hHat removes the excitation at the singularity, and
% the modification to the demoninator avoids errors due to irrelevant nans.
% Note that denominator is a nSPad-length column vector, but is implicitly 
% expanded to a nSPad x nTPad matrix for multiplication in the kZ equation.
denominator = (U*kS).^2 - fC^2;
iZero = abs(denominator)<(U*dKS/2)^2;
hHat(iZero) = 0;
clear iZero
denominator(denominator==0) = eps;

%... Calculate kZ, vertical wave number grid
kZ2 = (kS.^2 + kT.^2).*((NM.^2 - (U*kS).^2)./denominator) - 1/(4*hRho^2);
kZ = sqrt(kZ2);
%... Assign appropriate roots for sqrt(kZ2)
% If kZ2>0, then sqrt(kZ2) is real (propagating wave).
% The sign of real kZ values is set to match the sign of its associated 
% kS value, which ensures that the wave propagates upward for both 
% positive and negative wavenumbers for kS. 
% Note that the logical calculation for iNeg does an implicit expansion 
% of the nSPad-length row vector on the right.
iNeg = (kZ2>0) & (kS<0);
kZ(iNeg) = -kZ(iNeg);

end