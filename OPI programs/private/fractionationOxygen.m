function alpha = fractionationOxygen(T)
% Given temperature T in Kelvin, returns fractionation factors for
% oxygen isotopes for condensates water and ice relative to water vapor.
% The algorithm is based on the mixed cloud isotopic model
% (MCIM) of Ciais and Jouzel (1994), which accounts for supercooling
% required to nucleate ice, kinetic fractionation for vapor to ice,
% and Wegener-Bergeron-Findeisen (WBF) zone, defined by the temperature
% range 268 to 248 K where water and ice coexist.
% Fractionation equations below are defined relative to the following
% generic polynomial function:
%   alpha = exp( ...
%                blvp3 T^3 + blvp2 T^2 + blvp1 T ...
%                + blv0 ...
%                + blvm1 T^-1 + blvm2 T^-2 + blvm3 T^-3 )

% Mark Brandon, Yale University, 2016 - 2018

%% Initialize variables
% Kelvin to Celsius
TC2K = 273.15;

%% Ice-vapor exchange
%... Equilibrium fractionation factor for oxygen isotopes in
% ice and vapor for -33 to 0 C from Majoube, 1970, Nature.
bivp3 = 0;
bivp2 = 0;
bivp1 = 0;
biv0 =  -28.224e-3;
bivm1 = 11.839;
bivm2 = 0;
bivm3 = 0;
%... Equilibrium fractionation factor for ice-vapor exchange
alphaIV = exp( ...
    bivp3.*T.^3 + bivp2.*T.^2 + bivp1.*T.^1 ...
    + biv0 ...
    + bivm1.*T.^-1 + bivm2.*T.^-2 + bivm3.*T.^-3 );

%... Adjust fractionation for kinetic effect for ice and vapor,
% according to Ciais and Jouzel (1994).
sI = 1.02 - 0.0038*(T-TC2K);
%... Ratio diffusivity(HH18O) / diffusivity(HH16O) from Merlivat, 1978
diffusivityRatio = 0.9723;
alphaIV = alphaIV.*sI./((alphaIV.*(sI-1)./diffusivityRatio) + 1);

%% Water-vapor exchange
%... Equilibrium fractionation factor for oxygen isotopes in
% water and vapor for 273 to 373 K from Majoube, 1971 
% (as reported in Criss, 1999, p. 103).
blvp3 = 0;
blvp2 = 0;
blvp1 = 0;
blv0 = -2.0667e-3;
blvm1 = -0.4156;
blvm2 = 1.137e3;
blvm3 = 0;
%... Equilibrium fractionation factors for water-vapor exchange
alphaLV = exp( ...
    blvp3.*T.^3 + blvp2.*T.^2 + blvp1.*T.^1 ...
    + blv0 ...
    + blvm1.*T.^-1 + blvm2.*T.^-2 + blvm3.*T.^-3 );

%% Combine results
%... For mixed ice and water, factor goes from zero for T < 248 K
% to one for T > 268 K, corresponding to the WBF zone.
factor = (T-248)./(268-248);
factor(factor>1) = 1;
factor(factor<0) = 0;
alpha = alphaLV.*factor + alphaIV.*(1-factor);

end