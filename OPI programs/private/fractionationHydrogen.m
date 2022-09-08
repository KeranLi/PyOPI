function alpha = fractionationHydrogen(T)
% Given temperature T in Kelvin, returns fractionation factors for
% hydrogen isotopes for condensates water and ice relative to
% water vapor. The algorithm is based on the mixed cloud isotopic model
% (MCIM) of Ciais and Jouzel (1994), which accounts for supercooling
% required to nucleate ice, kinetic fractionation for vapor to ice,
% and Bergeron-Findeisen (WBF) zone, defined by the temperature
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

%% Ice-vapor calculation
%... Equilibrium fractionation factor for hydrogen isotopes
% in ice and vapor for -40 to 0 C from Merlivat and Nief, 1967, Tellus.
bivp3 = 0;
bivp2 = 0;
bivp1 = 0;
biv0 =  -9.45e-2;
bivm1 = 0;
bivm2 = 16289;
bivm3 = 0;
%... For ice-vapor only
alphaIV = exp( ...
    bivp3.*T.^3 + bivp2.*T.^2 + bivp1.*T.^1 ...
    + biv0 ...
    + bivm1.*T.^-1 + bivm2.*T.^-2 + bivm3.*T.^-3 );

%... Adjust fractionation for kinetic effect for ice and vapor,
% according to Ciais and Jouzel, 1994.
sI = 1.02 - 0.0038*(T-TC2K);
%... Ratio diffusivity(HD16O) / diffusivity(HH16O) from Merlivat, 1978
diffusivityRatio = 0.9755;
alphaIV = alphaIV.*sI./((alphaIV.*(sI-1)./diffusivityRatio) + 1);

%% Water-vapor calculation
% Water-vapor for hydrogen isotopes is represented by two studies,
% covering two different temperature ranges

%... Equilibrium fractionation factor for hydrogen isotopes
% in water and vapor for 273 to 373 K from Majoube, 1971
% (as reported in Criss, 1999, p. 103).
blvp3 = 0;
blvp2 = 0;
blvp1 = 0;
blv0 = 52.612e-3;
blvm1 = -76.248;
blvm2 = 24.844e3;
blvm3 = 0;
%... Equilibrium fractionation factors for water-vapor exchange
alphaLV_M71 = exp( ...
    blvp3.*T.^3 + blvp2.*T.^2 + blvp1.*T.^1 ...
    + blv0 ...
    + blvm1.*T.^-1 + blvm2.*T.^-2 + blvm3.*T.^-3 );

%... Equilibrium fractionation factor for hydrogen isotopes in
% water and vapor for 258 to 273 K from Merlivat and Nief, 1967, Tellus.
blvp3 = 0;
blvp2 = 0;
blvp1 = 0;
blv0 =  -10.0e-2;
blvm1 = 0;
blvm2 = 15013;
blvm3 = 0;
%... Equilibrium fractionation factors for water-vapor exchange
alphaLV_MN67 = exp( ...
    blvp3.*T.^3 + blvp2.*T.^2 + blvp1.*T.^1 ...
    + blv0 ...
    + blvm1.*T.^-1 + blvm2.*T.^-2 + blvm3.*T.^-3 );

%... Merge calibration results
iSwitch = (T>=TC2K);
alphaLV = iSwitch.*alphaLV_M71 + ~iSwitch.*alphaLV_MN67;

%% Combine results
%... For mixed ice and water, factor goes from zero for T < 248 K
% to one for T > 268 K, corresponding to the WBF zone.
factor = (T-248)./(268-248);
factor(factor>1) = 1;
factor(factor<0) = 0;
alpha = alphaLV.*factor + alphaIV.*(1-factor);
 
end
