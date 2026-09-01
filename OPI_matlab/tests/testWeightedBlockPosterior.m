function tests = testWeightedBlockPosterior
tests = functiontests(localfunctions);
end

function testZeroExternalWeightReturnsOwnBlock(testCase)
prior = [0.5; 0.5];
logBlocks = [log(0.8), log(0.1); log(0.2), log(0.9)];
posterior = weightedBlockPosterior(prior, logBlocks, [1; 0]);
verifyEqual(testCase, posterior, [0.8; 0.2], 'AbsTol', 1e-12);
end

function testIncreasingExternalWeightChangesPreference(testCase)
prior = [0.5; 0.5];
logBlocks = [log(0.8), log(0.1); log(0.2), log(0.9)];
low = weightedBlockPosterior(prior, logBlocks, [1; 0.25]);
high = weightedBlockPosterior(prior, logBlocks, [1; 1]);
verifyGreaterThan(testCase, low(1), high(1));
verifyEqual(testCase, sum(high), 1, 'AbsTol', 1e-12);
end
