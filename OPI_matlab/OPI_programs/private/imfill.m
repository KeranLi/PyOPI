function B = imfill(A, mode)
%IMFILL Local fallback for Image Processing Toolbox imfill.
%   B = IMFILL(A, 'holes') fills interior depressions in a numeric grid.
%   This lightweight implementation supports the OPI catchmentNodes use
%   case and is not a full replacement for Image Processing Toolbox imfill.

if nargin ~= 2 || ~ischar(mode) && ~isstring(mode) || ~strcmpi(string(mode), "holes")
    error('Local imfill fallback only supports imfill(A, ''holes'').')
end
if ~isnumeric(A) || ndims(A) ~= 2
    error('Input must be a 2-D numeric array.')
end

B = A;
if isempty(B) || any(size(B) < 3)
    return
end

% Iteratively raise strict interior pits to the lowest neighboring level.
% Boundary values are left unchanged so catchments can still drain outward.
[m, n] = size(B);
maxIter = m*n;
for iter = 1:maxIter
    changed = false;
    for i = 2:m-1
        for j = 2:n-1
            neighbors = [ ...
                B(i-1,j-1), B(i-1,j), B(i-1,j+1), ...
                B(i,  j-1),           B(i,  j+1), ...
                B(i+1,j-1), B(i+1,j), B(i+1,j+1)];
            spill = min(neighbors);
            if B(i,j) < spill
                B(i,j) = spill;
                changed = true;
            end
        end
    end
    if ~changed
        break
    end
end
end
