function [ijCatch, ptrCatch] = ...
    catchmentNodes(sampleX, sampleY, sampleLC, x, y, hSGrid)
% Find nodes in hSGrid that are upstream of each sample location, as
% given by sampleX, sampleY. sampleLC should have values of L or C, 
% indicating a local sample (no catchment needed), or a catchment sample
% (where an upslope catchment is needed). x,y are grid vectors for the 
% elevation grid hSGrid. The output arguments are the vector ijCatch, 
% which contains the linear indices for catchment nodes in hSGrid,
% and the nx1 vector ptrCatch, which contains a pointer for each
% of the n samples to the starting index in ijCatch for the catchment
% nodes of each sample.

%% Initialize variables
%#ok<*AGROW> % Supress preallocation warning
[m, n] = size(hSGrid);
nSamples = length(sampleX);
ijCatch = zeros(0, 1);
ptrCatch = zeros(nSamples, 1);
ptrCatch(1) = 1;
%... Initiate indices for D8 neighbors
iD8 = [ 0 -1 -1 -1  0  1  1  1]';
jD8 = [ 1  1  0 -1 -1 -1  0  1]';
% Fill sinks
hSGrid = imfill(hSGrid, 'holes');

%% Compute
%... Iterate through samples
for k = 1:nSamples
    % Initate vector for linear indices for sample catchment nodes
    ijCatchSample(1) = sub2ind([m,n], ...
        round(interp1(y, 1:m, sampleY(k), 'linear', 'extrap')), ...
        round(interp1(x, 1:n, sampleX(k), 'linear', 'extrap')));
    if sampleLC(k)=='L'
        % Local water sample (sampleLC == L), no catchment needed
        % Append one node for sample to full list of catchment nodes
        if k~=nSamples, ptrCatch(k+1) = ptrCatch(k) + 1; end
        ijCatch(ptrCatch(k),1) = ijCatchSample(1);        
        continue
    end
    % Calculate upslope nodes for catchment water sample (sampleLC == C)
    % Initiate logical array to track identified sample catchment nodes
    isCatchSample = false(m,n);
    isCatchSample(ijCatchSample(1)) = true;
    kC = 1; % Number of demonstrated upslope grid nodes for ith sample
    nC = 1; % Number of grid nodes to be tested for ith sample
    NC = 1; % Physical length of iCatchSample
    %... Iterate through potential catchment nodes
    while true
        % Indices for D8 neighbors
        [i0,j0] = ind2sub([m,n],ijCatchSample(kC));
        i = i0 + iD8;
        j = j0 + jD8;
        % Limit search to range of hSGrid
        isInside = i>0 & i<=m & j>0 & j<=n;
        ij = sub2ind([m,n],i(isInside),j(isInside));
        % Identify new upslope neighbors
        isUpslope = ~isCatchSample(ij) ...
            & (hSGrid(ij) >= hSGrid(ijCatchSample(kC)));
        ij = ij(isUpslope);
        lC = sum(isUpslope);
        % Increase size of vectors, if needed
        if (nC + lC) > NC
            NC = NC + 100;
            ijCatchSample(NC,1) = 0;
        end
        % Add upslope grid points to catchment vectors
        if lC>0
            ijCatchSample(nC+1:nC+lC) = ij;
            isCatchSample(ij) = true;
            nC = nC + lC;
        end
        % Check for termination, which occurs when there are
        % no more upslope grid points.
        if kC==nC
            ijCatchSample = ijCatchSample(1:nC);
            break
        end
        % Prepare for next loop
        kC = kC + 1;
    end
    %... Append results to full list of catchment nodes
    if k~=nSamples, ptrCatch(k+1) = ptrCatch(k) + nC; end
    ijCatch(ptrCatch(k) + (0:nC-1),1) = ijCatchSample;
end
end