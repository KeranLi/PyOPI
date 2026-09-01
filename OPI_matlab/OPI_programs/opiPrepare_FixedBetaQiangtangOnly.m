function runFile = opiPrepare_FixedBetaQiangtangOnly()
root='/Users/keranli/Desktop/Coding/OPI_matlab';
fullDir=fullfile(root,'scenarios/Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth/complete_joint_simulation/Q4_V35_G4/north_source_state2');
qDir=fullfile(root,'scenarios/Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth/complete_joint_simulation/Q4_V35_G4/qiangtang_only_500m');
runFile=fullfile(qDir,'Tibet_Eocene_30Ma_Q4_V35_G4_QiangtangOnly500m_FixedFullBeta.run');
txt=splitlines(string(fileread(fullfile(fullDir,'Tibet_Eocene_30Ma_Q4_V35_G4_northsource_Best.run'))));
txt(1)="% Fixed-beta terrain contrast: full Q4/V3.5/G4 beta on Qiangtang-only topography";
txt(3)="Qiangtang-only 500 m terrain with fixed full Q4/V3.5/G4 beta";
txt(5)=string(qDir); txt(7)="Tibet_Eocene_30Ma_topo.mat";
txt(9)="Tibet_Eocene_30Ma_samples.xlsx"; txt(10)="Tibet_Eocene_30Ma_topo_divide_main.mat";
fid=fopen(runFile,'w'); fprintf(fid,'%s\n',txt); fclose(fid);
fprintf('Wrote %s\n',runFile);
end
