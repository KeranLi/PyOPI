function opiSetup_ClumpedSensitivityCases_ParameterGroup()
% opiSetup_ClumpedSensitivityCases_ParameterGroup builds all first-batch
% local-parameter sensitivity case skeletons.

opiSetup_ClumpedSensitivityCases("", "parameter");

fprintf(['Created first-batch parameter sensitivity cases under:\n' ...
    'sensitivity_parameter_local_clumped\n']);
end
