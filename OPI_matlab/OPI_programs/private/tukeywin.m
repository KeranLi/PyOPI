function w = tukeywin(n, r)
%TUKEYWIN Local fallback for Signal Processing Toolbox tukeywin.
%   Supports the OPI use case: a column-vector Tukey window of length n
%   with taper ratio r in [0, 1].

if nargin < 2 || isempty(r)
    r = 0.5;
end
if ~(isscalar(n) && isnumeric(n) && isfinite(n) && n == floor(n) && n >= 0)
    error('Window length must be a nonnegative integer scalar.')
end
if ~(isscalar(r) && isnumeric(r) && isfinite(r))
    error('Taper ratio must be a finite numeric scalar.')
end

n = double(n);
r = max(0, min(1, double(r)));

if n == 0
    w = zeros(0, 1);
    return
end
if n == 1 || r <= 0
    w = ones(n, 1);
    return
end
if r >= 1
    idx = (0:n-1)';
    w = 0.5 - 0.5*cos(2*pi*idx/(n-1));
    return
end

idx = (0:n-1)';
x = idx/(n-1);
w = ones(n, 1);

lower = x < r/2;
upper = x >= 1 - r/2;

w(lower) = 0.5*(1 + cos(2*pi/r*(x(lower) - r/2)));
w(upper) = 0.5*(1 + cos(2*pi/r*(x(upper) - 1 + r/2)));
end
