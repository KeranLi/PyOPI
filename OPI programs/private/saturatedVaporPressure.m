function eS = saturatedVaporPressure(T)
%... Saturated vapor pressure over water and ice, from
% Goff and Gratch (1946), as modified by Goff (1965).
% See review by Murphy and Koop (2005) for details. 
% The Wegener-Bergeron-Findeisen (WBF) zone, defined by the 
% temperature range 268 to 248 K where water and ice coexist.
% For this zone, eS is set using weighted averages for water
% and ice components. 
%
% Arguments: 
% Inputs can be scalar, vector, or array, and are matched with outputs. 
% T = temperature (K)
% eS = saturated vapor pressure (Pa)
% Outputs:
% eS = vapor pressure (Pa)

% Mark Brandon, Yale University August, 2020

%% Compute
%... Saturated vapor pressure (Pa) in equilibrium with water
eS_Water = 10.^( ...
    -7.90298*(373.16./T - 1) ...
    + 5.02808*log10(373.16./T) ...
    - 1.3816e-7*(10.^(11.344*(1 - T./373.16)) - 1) ...
    + 8.1328e-3*(10.^(3.49149*(1 - 373.16./T)) - 1) + log10(101324.6) );
%... Saturation vapor pressure (Pa) in equilibrium with ice
eS_Ice = 10.^( ...
    -9.09718*((273.16./T)-1) ...
    - 3.56654*log10(273.16./T) ...
    + 0.876793*(1 - (T/273.16)) + log10(610.71));

%... Mix values using a factor that goes from zero for T < 248 K
% to one for T > 268 K, corresponding to the WBF zone.
factor = (T-248)/(268-248);
factor(factor<0) = 0; % All ice
factor(factor>1) = 1; % All water
eS = eS_Ice.*(1-factor) + eS_Water.*factor;

end