function [zBar, T, gammaEnv, gammaSat, gammaRatio, ...
    rhoS0, hS, rho0, hRho] = baseState(NM, T0)
%... baseState returns variables for the state of the atmosphere 
% in the absense of topography. The top of the profile is set to zMax,
% with a recommended value of 12 km.
%
%... Input arguments:
% T0 = sea-level temperature (scalar, K)
% NM = saturated buoyancy frequency (scalar, rad/s)
%
%... Output arguments:
% zBar = elevation relative to sea level (vector, m)
% T = environmental temperature (vector, K)
% gammaEnv = environmental lapse rate (vector, K/m)
% gammaSat = saturation lapse rate (vector, K/m)
% gammaRatio = weight mean of gammaSat/gammaEnv
% rhoS0 = water-vapor density at z=0 (scalar, kg/m^3)
% hS = scale height for saturated vapor (scalar, m)
% rho0 = total air density at z=0 (scalar, kg/m^3)
% hRho = scale height for total air density (scalar, m)
%
%... This calculation solves for the vertical structure of 
% the base state for a saturated atmosphere with a specified 
% surface temperature, T0, and a constant buoyancy frequency, NM. 

% The saturated-vapor density profile is characterized by an 
% exponential relationship, rhoS(z) = rhoS0*exp(-z/hS). The parameters,
% rhoS0 and hS, are determined by a log-transformed least-squares fit. 
% Weights are set to be proportional to the distribution of  water vapor, 
% and to account for the log transformation used to linearize the fit. 
%
% The total-air density profile is characterized by an exponential 
% relationship, rho(z) = rho0*exp(-z/hRho). The parameters,
% rhoS0 and hS, are determined by a log-transformed least-squares fit. 
% Weights are set to account for the log transformation used to 
% linearize the fit.
%
%... Measures of humidity (see AMS Glossary of Meteorology)
% The variables below with subscript V are general for both 
% unsaturated and saturated conditions.
% Mixing ratio, rV (vapor mass/dry air mass):
%    rV = epsilon*eV/(p - eV), or rV = qV/(1 - qV).
% Vapor pressure, eV (Pa): 
%    eV = p*rV/(epsilon + rV)
% Specific humidity, q (vapor mass/total mass):
%    qV = rV/(1 + rV) = rhoV/rhoTotal.
% Water-vapor density rhoV (kg/m^3): 
%    rhoV = p*rV/(RD*T*(1 + rV/epsilon)).
% Total density rhoTotal (kg/m^3):
%    rhoTotal = p*(1 + rTotal)/(RD*T*(1 + rS/epsilon)
%
% Dependencies: saturatedVaporPressure
% 
% Mark Brandon, Yale University, 2016-2020

%% Check input arguments
if nargin==0, error('Input arguments are missing'); end

%% Constants
g = 9.81;           % standard gravity (m/s^2)
cPD = 1005.7;       % heat capacity dry air, constant pressure & 0 C (J/kg-K)
cPV = 1952;         % heat capacity water vapor, constant pressure & 0 C (J/kg-K)
RD = 287.0;         % specific gas constant dry air (J/kg-K)
L = 2.501e6;        % latent heat of vaporization (water -> vapor) (J/kg)
p0 = 101325;        % standard sea-level pressure (Pa)
epsilon = 0.622;    % molecular mass of water relative to air (kg/kg)
dZ = 100;           % elevation increment (m) (recommended: 100 m)
zMax = 12e3;        % maximum elevation (m)

%% Compute base state
n = round(zMax/dZ) + 1;
zBar = (0:n-1)'*dZ;
T = zeros(n,1);
T(1) = T0;
p = zeros(n,1);
p(1) = p0;
rS = zeros(n,1);
gammaEnv = zeros(n,1);
gammaSat = zeros(n,1);
rho = zeros(n,1);

%... Define nested function for numerically solving for gammaEnv
function diff = fDiff(gammaEnvEst)
    % Vertical derivative of saturation mixing ratio (DK82, eq. 12)
    dRS_dZ = rS(i)*(1 + rS(i)/epsilon)/(RD*T(i)) ...
        *(-epsilon.*L.*gammaEnvEst/T(i) ...
        + g.*(1 + rS(i))./(1 + rS(i)./epsilon));
    %... Saturated buoyancy frequency using DK82, eq 5, with rT = rS
    NS2Est = (g/T(i))*(gammaSat(i) - gammaEnvEst) ...
        *(1 + L*rS(i)/(RD*T(i))) - g/(1 + rS(i)) *dRS_dZ;
    diff = NM^2 - NS2Est;    
end
%... End of nested function

%... Calculate profile by integrating upward from sea level.
for i = 1:n
    %... Partial pressure and mixing ratio for saturated water vapor
    eS = saturatedVaporPressure(T(i));
    rS(i) = epsilon*eS/(p(i) - eS);
    % Saturated adiabatic lapse rate, DK82, eq. 19, with 2`Q
    gammaSat(i) = g*(1 + rS(i))*(1 + L*rS(i)/(RD*T(i))) ...
        /(cPD + cPV*rS(i) + L^2*rS(i)*(epsilon + rS(i))/(RD*T(i)^2));
    %... Use fzero to get gammaEnv using eq. 5 from Durran and Klemp, 1982
    % Start with guess based on eq. 3 from Durran and Klemp, 1982
    gammaEnvGuess = gammaSat(i) - NM^2*T(i)/g;
    gammaEnv(i) = fzero(@fDiff, gammaEnvGuess);
    % Total density, eq. 3.15 in Wallace and Hobbs, 2006
    rho(i) = p(i)*(1 + rS(i))/(RD*T(i)*(1 + rS(i)/epsilon));    
    %... Forward difference for temperature, pressure, and 
    % saturated adiabat at next elevation 
    if i<n
        T(i+1) = T(i) - gammaEnv(i)*dZ;
        dLnP_dZ = -g*rho(i)/p(i);        
        p(i+1) = p(i).*exp(dLnP_dZ*dZ);
    end
end    
%... Check gammaEnv
if any(gammaEnv<1e-3)
    error(['Not allowed: Selected NM and T0 for the base state ', ...
        'local regions where gammaEnv < 1 C/km.'])
end

%... Calculate water-vapor density
rhoS = p.*rS./(RD.*T.*(1 + rS./epsilon));

%... Estimate gammaRatio, which is the mean of the 
% ratio of the saturated lapse rate over the environmental lapse rate. 
% The mean is weighted proportional to the saturated-vapor density.
weights = rhoS./sum(rhoS);
gammaRatio = sum(weights.*gammaSat./gammaEnv);

%... Estimate rhoS0 and hS for vertical water-vapor density. 
% These parameters are estimated using an exponential fit, with 
% weighting set proportional to the saturated-vapor density, and 
% to account for the log transformation used for the fit.
A = [ones(n,1), zBar];
weights = rhoS.^2/sum(rhoS.^2); 
b = weights.*A\(weights.*log(rhoS));
rhoS0 = exp(b(1));
hS = -1/b(2);

%... Exponential fit to estimate rho0 and hRho for total density
% Total density is based on eq. 3.15 in Wallace and Hobbs, 2006.
% Weights account for log transformation used for the fit. 
rhoTotal = p.*(1 + rS)./(RD.*T.*(1 + rS./epsilon));
A = [ones(n,1), zBar];
weights = rhoTotal/sum(rhoTotal); 
b = weights.*A\(weights.*log(rhoTotal));
rho0 = exp(b(1));
hRho = -1/b(2);

%... Create WRF sounding file
% U = 10; % m/s
% fid1 = fopen('WRF_initialSounding.txt', 'w');
% thetaDry = T.*(p0./p).^0.286;
% fprintf(fid1, '%f\t%f\t%f\n', p0*1e-2, thetaDry(1), rS(1)*1e3);
% for i = 1:n
%     fprintf(fid1, '%f\t%f\t%f\t%f\t%f\n', zBar(i), thetaDry(i), rS(i)*1e3, U, 0);
% end
% fclose(fid1);
% fprintf('WRF initial sounding saved at:\n')
% fprintf('%s\n', [pwd, '/', 'WRF_initialSounding.txt'])

end
