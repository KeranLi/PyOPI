function zR = isotherm(TR, zBar, T, gammaEnv, gammaSat, hRho, ...
    nS, nT, hHat, kZ)
%... Calculate height array for isothermal surface with temperature TR.
% Input arguments:
% TR = specified isotherm temperature (scalar, K)
% zBar = base-state elevation vector for base-state variables (vector, m)
% T = base-state temperature relative to zBar (vector, K)
% gammaEnv = base-state environmental lapse rate (vector, K/m)
% gammaSat = base-state saturation lapse rate (vector, K/m)
% hRho = base-state scale height for air density (scalar, m)
% nS, nT = row and column dimensions for hHat
% hHat = Fourier coefficients for topography (matrix, m)
% kZ = vertical wave number (matrix, rad/m)
%
% Output arguments:
% zR = height of TR surface (matrix, m)
%
% Note that isotherm will do a linear extrapolation of the 
% base-state profile if TR is outside of the range of that profile.

% Mark Brandon, Yale University 2021

%% Compute
% Base-state elevation of specified isotherm temperature TR
zBarR = interp1(T, zBar, TR, 'linear', 'extrap');
% Base-state environmental lapse rate
gammaEnvBarR = interp1(zBar, gammaEnv, zBarR,'linear', 'extrap');
% Base-state saturation lapse rate
gammaSatBarR = interp1(zBar, gammaSat, zBarR, 'linear', 'extrap');
% Calculate perturbation height, zPrime, for the zBarR stream surface
zPrime = ifft2(hHat.*exp((1i.*kZ + 1/(2*hRho))*zBarR), 'symmetric');
zPrime = zPrime(1:nS, 1:nT);
zR = zBarR + zPrime*(1 - gammaSatBarR/gammaEnvBarR);

end
