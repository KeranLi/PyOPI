function h = refline(slope, intercept)
%REFLINE Local fallback for Statistics Toolbox refline.
%   h = REFLINE(slope, intercept) adds y = slope*x + intercept to axes.

if nargin == 0
    slope = 1;
    intercept = 0;
elseif nargin == 1
    if numel(slope) == 2
        intercept = slope(2);
        slope = slope(1);
    else
        intercept = 0;
    end
end

ax = gca;
xLim = ax.XLim;
y = slope*xLim + intercept;
h = line(ax, xLim, y);
end
