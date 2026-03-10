function ij = catchmentIndices(k, ijCatch, ptrCatch)
% Extract indices for sample catchment
% Input arguments:
% k: index for selected catchment (integer, scalar)
% ijCatch: linear indices for catchment as represented by grid
%    nodes in hGrid (integer, vector)
% ptrCatch: pointers for first node of each catchment as 
%    represented in ijCatch (integer, vector). 

%% Compute
n_ijCatch = length(ijCatch);
nSamples = length(ptrCatch);
if k~=nSamples
    ij = ijCatch(ptrCatch(k):ptrCatch(k+1)-1);
else
    ij = ijCatch(ptrCatch(k):n_ijCatch);
end

end