function [ijCatch, ptrCatch] = catchmentNodes_fixed(sampleX, sampleY, sampleLC, x, y, hSGrid)
% Find nodes in hSGrid that are upstream of each sample location
% Fixed version with bounds checking

%% Initialize variables
[m, n] = size(hSGrid);
nSamples = length(sampleX);
ijCatch = zeros(0, 1);
ptrCatch = zeros(nSamples, 1);
if nSamples > 0
    ptrCatch(1) = 1;
end

% D8 neighbor indices
iD8 = [ 0 -1 -1 -1  0  1  1  1]';
jD8 = [ 1  1  0 -1 -1 -1  0  1]';

% Fill sinks
hSGrid = imfill(hSGrid, 'holes');

%% Compute
for k = 1:nSamples
    % Calculate row and column indices with bounds checking
    rowIdx = round(interp1(y, 1:m, sampleY(k), 'linear', 'extrap'));
    colIdx = round(interp1(x, 1:n, sampleX(k), 'linear', 'extrap'));
    
    % CRITICAL: Ensure indices are within valid range [1, m] and [1, n]
    rowIdx = max(1, min(m, rowIdx));
    colIdx = max(1, min(n, colIdx));
    
    % Calculate linear index
    ijCatchSample(1) = sub2ind([m,n], rowIdx, colIdx);
    
    if sampleLC(k)=='L'
        % Local water sample
        ijCatch = [ijCatch; ijCatchSample(1)];
        if k~=nSamples, ptrCatch(k+1) = ptrCatch(k) + 1; end
        continue
    end
    
    % Catchment sample - calculate upslope nodes
    isCatchSample = false(m,n);
    isCatchSample(ijCatchSample(1)) = true;
    kC = 1;
    nC = 1;
    NC = 1;
    
    while true
        [i0,j0] = ind2sub([m,n],ijCatchSample(kC));
        i = i0 + iD8;
        j = j0 + jD8;
        isInside = i>0 & i<=m & j>0 & j<=n;
        ij = sub2ind([m,n],i(isInside),j(isInside));
        isUpslope = ~isCatchSample(ij) & (hSGrid(ij) >= hSGrid(ijCatchSample(kC)));
        ij = ij(isUpslope);
        lC = sum(isUpslope);
        
        if (nC + lC) > NC
            NC = NC + 100;
            ijCatchSample(NC,1) = 0;
        end
        
        if lC>0
            ijCatchSample(nC+1:nC+lC) = ij;
            isCatchSample(ij) = true;
            nC = nC + lC;
        end
        
        if kC==nC
            ijCatchSample = ijCatchSample(1:nC);
            break
        end
        kC = kC + 1;
    end
    
    if k~=nSamples, ptrCatch(k+1) = ptrCatch(k) + nC; end
    ijCatch = [ijCatch; ijCatchSample];
end
end
