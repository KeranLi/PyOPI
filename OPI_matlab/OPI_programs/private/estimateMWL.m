function [bHat, sdMin, sdMax, cov, iFit] = estimateMWL(d18O, d2H, sdResRatio)
% Estimate meteoric water line (MWL) using total least squares.
% The data are trimmed to remove samples with dExcess < 5 per mil,
% which works well to identify precipitation samples that have been 
% altered by evaporation. The total least sqares method is used 
% because the covariance for the isotope data has principal directions
% that are aligned with the MWL. 
% Input:
% d18O, d2H data (per unit, or per mil).
% sdResRatio: Estimated standard-deviation ratio for isotopic variation at 
% a location due to seaonal variation, where the ratio equals
% maximum sd/minimum sd.
% Output:
% bHat: coeficients for best fit, where d2H = bHat(1) + bHat(2)*d18O.
% sdMin, sdMax: minimum and maximum standard deviation, where sdMin is
%    the standard deviation of the residuals normal to the MWL.
% cov: estimated covariance for isotopes due to seasonal variation. 
% iFit: logical vector indicating samples used for fit.

% Mark Brandon, Yale University  2018-2021

%% Compute
%... Identify samples with dExcess < 5 per mil.
dExcess = d2H - 8*d18O;
iFit = dExcess > 5e-3;
nFit = sum(iFit);

% Estimate a straight-line fit using total least squares
d2HMean = mean(d2H(iFit));
d18OMean = mean(d18O(iFit));
A = [d18O(iFit) - d18OMean, d2H(iFit) - d2HMean];
%... Eigen decomposition of covariance matrix
[V, D] = eig(A'*A/(nFit - 1));
[D, iSort] = sort(diag(D));
V = V(:,iSort);
sdMin = sqrt(D(1));
sdMax = sqrt(D(2));
%... Slope of MWL  = V(2,2)/V(1,2) (components of major eigenvector),
% and  (d2H - d2HMean) = slope*(d18O - d18OMean)
% Recast as: d2H = d2HMean - slope*d18OMean + slope*d18O;
bHat(2) = V(2,2)/V(1,2);
bHat(1) = d2HMean - bHat(2)*d18OMean;

%... Create estimated covariance matrix for isotope residuals, and
% organize so that V(d2H) and V(d18O) are 1st and 2nd on the diagonal,
% which is required for the chiR2 calculation in calc_OneWind 
% and calc_TwoWinds functions.
cov = V*diag([D(1), sdResRatio^2*D(1)])*V';
cov = [cov(2,2) cov(1,2); cov(2,1), cov(1,1)];

end
