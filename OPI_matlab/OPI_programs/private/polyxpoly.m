function [xi, yi, ii] = polyxpoly(x1, y1, x2, y2)
%POLYXPOLY Local fallback for Mapping Toolbox polyxpoly.
%   Supports the OPI use case: intersections between two polylines.
%   ii(k,:) contains segment indices [i1 i2] for the kth intersection.

x1 = x1(:); y1 = y1(:); x2 = x2(:); y2 = y2(:);
if numel(x1) ~= numel(y1) || numel(x2) ~= numel(y2)
    error('Polyline coordinate vectors must have matching lengths.')
end

xi = zeros(0,1);
yi = zeros(0,1);
ii = zeros(0,2);

for i = 1:numel(x1)-1
    p = [x1(i), y1(i)];
    r = [x1(i+1) - x1(i), y1(i+1) - y1(i)];
    for j = 1:numel(x2)-1
        q = [x2(j), y2(j)];
        s = [x2(j+1) - x2(j), y2(j+1) - y2(j)];
        denom = cross2(r, s);
        qp = q - p;
        if abs(denom) < eps(max([1, abs(r), abs(s)]))
            continue
        end
        t = cross2(qp, s) / denom;
        u = cross2(qp, r) / denom;
        if t >= 0 && t <= 1 && u >= 0 && u <= 1
            point = p + t*r;
            xi(end+1,1) = point(1); %#ok<AGROW>
            yi(end+1,1) = point(2); %#ok<AGROW>
            ii(end+1,:) = [i, j]; %#ok<AGROW>
        end
    end
end
end

function z = cross2(a, b)
z = a(1)*b(2) - a(2)*b(1);
end
