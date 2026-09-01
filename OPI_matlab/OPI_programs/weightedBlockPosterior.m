function posterior = weightedBlockPosterior( ...
    priorProbability, logLikelihoodBlocks, blockWeights)
% Combine independent likelihood blocks using explicit power weights.


priorProbability = priorProbability(:);
blockWeights = blockWeights(:);
if size(logLikelihoodBlocks, 1) ~= numel(priorProbability) || ...
        size(logLikelihoodBlocks, 2) ~= numel(blockWeights) || ...
        any(~isfinite(priorProbability)) || any(priorProbability < 0) || ...
        sum(priorProbability) <= 0 || any(~isfinite(blockWeights)) || ...
        any(blockWeights < 0)
    error('Invalid prior, likelihood-block matrix, or block weights.');
end
priorProbability = priorProbability ./ sum(priorProbability);
logWeight = log(priorProbability) + logLikelihoodBlocks * blockWeights;
finite = isfinite(logWeight);
if ~any(finite)
    error('No finite weighted likelihood cases remain.');
end
posterior = zeros(size(logWeight));
maximum = max(logWeight(finite));
posterior(finite) = exp(logWeight(finite) - maximum);
posterior = posterior ./ sum(posterior);
end
