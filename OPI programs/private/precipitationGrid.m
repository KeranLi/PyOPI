function [s, t, Sxy, Txy, pGrid, hWind, fMWind, rHWind, fPWind, ...
    z223Wind, z258Wind, tauF] = precipitationGrid ...
    (x, y,  hGrid, U, azimuth, NM, fC, kappa, tauC, hRho, zBar, ...
    T, gammaEnv, gammaSat, gammaRatio, rhoS0, hS, fP0)
%... precipitationGrid calculates grids for precipitation rate 
% (kg m^-2 s^-1) and moisture ratio (dimensionless), using a modified 
% version of the LTOP algorithm of Smith and Barstad (2004). 
% The modifications include Coriolis forcing, and moisture balance. 
% The function assumes that all grids are in the "wind grid" format, 
% with coordinates s,t corresponding to the 1st and 2nd dimensions 
% of the grid, respectively. The +s axis points in the downwind 
% direction, and the +t axis is 90 degrees clockwise for the +s axis.
%
% Feb 2021: Added moisture balance including evaporative recycling,
% where fP is the fraction of precipitation that leaves the 
% base of the model.

% August 8, 2021: Evaporation calculation had an error, now fixed, where fP 
% was left everywhere equal to 1, rather than being set to the candidate 
% value, fP0, prescribed for each step of the search. This error has 
% likely been active since March 23, 2021 (OPI 3.5).

% Mark Brandon, Yale University, 2016-2021

%% Initialize system
warning('off', 'MATLAB:griddedInterpolant:MeshgridEval2DWarnId');

%% Constants
% Dry adiabatic lapse rate (K/m)
gammaDry = 0.009754;

%% Compute reference solution
%... Get Fourier solution for Euler equations
 [s, t, Sxy, Txy, hWind, kS, kT, hHat, kZ] = ...
    fourierSolution(x, y, hGrid, U, azimuth, NM, fC, hRho);

%... Parameters for wind grids
dS = s(2)-s(1);
nS = length(s);
nT = length(t);

%... Calculate height array for the freezing surface as needed to 
% calculate tauF. The freezing surface is defined by the 
% 258 K isosurface, which marks the midpoint of the 268 - 248 K range
% for freezing in the atmosphere (WBF zone, Cias and Jouzel, 1994).
z258Wind = isotherm(258, zBar, T, gammaEnv, gammaSat, hRho, nS, nT, hHat, kZ);

%... Calculate mean fall time
% Velocities wR and wS (m/s) for rain and snow, respectively.
% References: Langleben, 1954; White et al., 2002; Yuter and Houze, 2003;
% Barstad and Schuller, 2011.
wFSnow = -1;
wFRain = -6;
% Mean fall time
tauF = (z258Wind<=hWind).* -hS/wFSnow ...
    + (z258Wind>hWind).* ...
    -((z258Wind - hWind)/wFRain + hS*exp(-(z258Wind - hWind)./hS)/wFSnow);
tauF = mean(tauF, 'all');
if isnan(tauF), stop, end
clear z258Wind

%... Calculate grid z223Wind, which is elevation relative to sea level of the 
% 223 K isothermal surface. Used by isotopeGrid function. 
z223Wind = ...
    isotherm(223, zBar, T, gammaEnv, gammaSat, hRho, nS, nT, hHat, kZ);

%... Calculate grid z258Wind, which is elevation relative to sea level of the 
% 258 K isothermal surface. Used by isotopeGrid function. 
z258Wind = ...
    isotherm(258, zBar, T, gammaEnv, gammaSat, hRho, nS, nT, hHat, kZ);

%... Calculate PStarHat, reference precipitation rate for wave domain
GSHat = gammaRatio*rhoS0*1i*kS.*U./(1 - hS*(1i*kZ + 1/(2*hRho)));
GCHat = 1./(tauC*(kappa*(kS.^2 + kT.^2) + 1i*kS*U) + 1);
GFHat = 1./(tauF*(kappa*(kS.^2 + kT.^2) + 1i*kS*U) + 1);
pStarHat = GSHat.*GCHat.*GFHat.*hHat;

%... Transform back to space domain, remove padding, and 
% set negative values in pStarPosWind to zero. The 'symmetric" option
% for ifft2 indicates that PStarHat is conjugate symmetric, which ensures
% that pStarWind is returned as a real-valued grid.
pStarPosWind = ifft2(pStarHat, 'symmetric');
clear pStarHat
pStarPosWind = pStarPosWind(1:nS, 1:nT);
pStarPosWind(pStarPosWind<0) = 0;

%% Calculate vapor ratio and moisture-corrected precipitation rate
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

QTStarPosWind = rhoS0*hS + QCStarPosWind + QFStarPosWind;
clear QCStarPosWind QFStarPosWind

%... Account for evaporatiive recycling
if fP0==1
        %... No evaporative recycling
        rHWind = 1;
        fPWind = 1;
else
        %... Evaporative recycling is restricted to areas where the air
        % at the base of the model is undersaturated, which is usually
        % above the lee slopes of the topography.
        % Start by calculating the cloud-water density, rhoC, at zBar = 0.
        % Note that rhoC is calculated in the LTOP model assuming 
        % that a parcel follows a moist-adiabat, with rhoC>=1 when
        % saturated, and rhoC<1 when undersaturated. 
        rhoCStarHat = (gammaRatio*rhoS0.*1i.*kS*U/hS).*hHat ...
            ./((kappa*(kS.^2 + kT.^2) + 1i*kS*U) + 1/tauC);
        rhoCStarWind = ifft2(rhoCStarHat, 'symmetric');
        rhoCStarWind = rhoCStarWind(1:nS, 1:nT);
        clear rhoCStarHat
        % Calculate relative humidity, with range 0 to 1. 
        % The ratio gammaDry/gammaSat(1) corrects for the fact that
        % rhoC follows the dry adiabat when undersaturated. Also note 
        % that rhoC and rhoS0 are both scaled by fV, so fV cancels out.
        rHWind = 1 + (gammaDry/gammaSat(1))*rhoCStarWind./rhoS0;
        rHWind(rHWind>1) = 1;
        % clear rhoCStarWind 
        % The residual precipitation grid, fPWind, is set to 1
        % where rHWind=1 (saturated), and to the specified fP0 
        % where rhWind<1.
        fPWind = ones(nS,nT);
        fPWind(rHWind<1) = fP0;
        
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         % For setting leeside relative humidity to a constant value.
%         % Chirp sound and print output warn that this kludge is active. 
%         S = load('chirp.mat'); sound(S.y)
%         rHWind0 = 1;
%         rHWind(rHWind<1) = rHWind0;
%         fprintf('\n>>> precipitationGrid: Manually set rHWind in leeside regions. <<<\n')
%         fprintf('rHWind0 = %g\n', rHWind0)
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
end

%... Integrate along the columns of the s,t grid (+s direction) to
% calculate the water-vapor ratio, fV.
fVWind = (rhoS0*hS./QTStarPosWind) ...
    .*exp(-(1/U)*cumtrapz(fPWind.*pStarPosWind.*dS./QTStarPosWind));

%... Calculate PGrid, moisture-corrected precipitation-rate field
pWind = fVWind.*pStarPosWind;
clear pStarPosWind

%... Transform precipitation rate back to geographic grid
F = griddedInterpolant({s, t}, pWind, 'linear', 'none');
pGrid = F(Sxy, Txy);
clear pWind

%... Calculate fM, moisture-ratio field
fMWind = fVWind.*QTStarPosWind/(rhoS0*hS);
clear QTStarPosWind fVWind

end