function result = run_assimilation(experimentRoot)
% Run OPI proxy data assimilation for a supplied experiment directory.
% The experiment must contain design/case_manifest.csv and calc_only/.
if nargin < 1 || isempty(experimentRoot)
    error('Pass experimentRoot containing design/case_manifest.csv and calc_only/.');
end
matlabRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(matlabRoot, 'OPI_programs'));
result = runCollectedCarbonateAssimilation(matlabRoot, experimentRoot);
end
