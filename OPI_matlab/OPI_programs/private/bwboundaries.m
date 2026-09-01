function [boundaries, labels, nObjects] = bwboundaries(BW)
%BWBOUNDARIES Local fallback for Image Processing Toolbox bwboundaries.
%   Supports the OPI plotting use case for 2-D logical masks.

if ~islogical(BW)
    BW = logical(BW);
end
if ndims(BW) ~= 2
    error('Input must be a 2-D array.')
end

labels = [];
boundaries = {};
if ~any(BW, 'all')
    nObjects = 0;
    return
end

% Pad with false values so regions touching the array edge have contours.
padded = padarray_local(BW, [1 1], false);
C = contourc(double(padded), [0.5 0.5]);

k = 1;
while k < size(C, 2)
    n = C(2, k);
    pts = C(:, k+1:k+n);
    % Convert contour x/y back to bwboundaries-style row/column indices.
    rows = round(pts(2,:)' - 1);
    cols = round(pts(1,:)' - 1);
    rows = max(1, min(size(BW,1), rows));
    cols = max(1, min(size(BW,2), cols));
    isInside = rows >= 1 & rows <= size(BW,1) & cols >= 1 & cols <= size(BW,2);
    rows = rows(isInside);
    cols = cols(isInside);
    if ~isempty(rows)
        boundaries{end+1,1} = [rows, cols]; %#ok<AGROW>
    end
    k = k + n + 1;
end

nObjects = numel(boundaries);
end

function B = padarray_local(A, padSize, padValue)
B = padValue(ones(size(A) + 2*padSize));
B(1+padSize(1):padSize(1)+size(A,1), ...
  1+padSize(2):padSize(2)+size(A,2)) = A;
end
