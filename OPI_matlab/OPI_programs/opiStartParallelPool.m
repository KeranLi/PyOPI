function pool = opiStartParallelPool(nWorkers)
% Start or reuse a local Parallel Computing Toolbox process pool.
% With no argument, use a conservative worker count for Apple Silicon.

if nargin < 1 || isempty(nWorkers)
    nWorkers = max(1, min(feature('NumCores') - 1, 8));
end
validateattributes(nWorkers, {'numeric'}, {'scalar', 'integer', 'positive'});
if isempty(ver('parallel')) || ~license('test', 'Distrib_Computing_Toolbox')
    error(['Parallel Computing Toolbox is required. Install it and verify ', ...
        'with ver parallel before starting a pool.']);
end

pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('Processes', nWorkers);
elseif pool.NumWorkers ~= nWorkers
    delete(pool);
    pool = parpool('Processes', nWorkers);
end
fprintf('OPI parallel pool ready: %d process workers.\n', pool.NumWorkers);
end
