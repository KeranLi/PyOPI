function x = tinv(p, nu)
%TINV Local fallback for Statistics Toolbox tinv.
%   Supports inverse Student's t CDF for positive degrees of freedom.

if ~isnumeric(p) || ~isnumeric(nu)
    error('Inputs must be numeric.')
end

[p, nu] = ndgrid(p, nu);
x = nan(size(p));

isValid = p >= 0 & p <= 1 & nu > 0 & isfinite(nu);
x(p == 0 & isValid) = -Inf;
x(p == 1 & isValid) = Inf;
x(p == 0.5 & isValid) = 0;

isLower = isValid & p > 0 & p < 0.5;
isUpper = isValid & p > 0.5 & p < 1;

if any(isLower, 'all')
    z = betaincinv(2*p(isLower), nu(isLower)/2, 0.5);
    x(isLower) = -sqrt(nu(isLower).*(1./z - 1));
end

if any(isUpper, 'all')
    z = betaincinv(2*(1 - p(isUpper)), nu(isUpper)/2, 0.5);
    x(isUpper) = sqrt(nu(isUpper).*(1./z - 1));
end
end
