function caseDir = opiPrepare_ModernTibetCase()
% Prepare a modern-Tibet OPI calculation case on the active ancient grid.
root = '/Users/keranli/Desktop/Coding/OPI_matlab';
anc = fullfile(root,'scenarios/Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth/topography_north_south_grid_smooth/calc_only/G4000_Q3500_V4000');
caseDir = fullfile(root,'scenarios/modern_Tibet_reference');
if ~isfolder(caseDir), mkdir(caseDir); end
S=load(fullfile(anc,'opiCalc_TwoWinds_OxygenOnly_Results.mat'),'lon','lat');
H=load(fullfile(root,'data/modern_Tibet/Himalaya topography_April 2023_Gebco1kmMakima.mat'),'lon','lat','hGrid');
[LON,LAT]=meshgrid(S.lon,S.lat);
hGrid=interp2(H.lon,H.lat,H.hGrid,LON,LAT,'linear'); hGrid(~isfinite(hGrid))=0; hGrid=max(hGrid,0);
lon=S.lon; lat=S.lat; save(fullfile(caseDir,'Tibet_Modern_topo.mat'),'lon','lat','hGrid');
copyfile(fullfile(anc,'Tibet_Eocene_30Ma_topo_divide_main.mat'), ...
    fullfile(caseDir,'Tibet_Modern_topo_divide_main.mat'));
T=readtable(fullfile(root,'data/modern_Tibet/Himalaya Water Isotopes 8 March 2023.xlsx'));
keep=T.Longitude>=min(S.lon)&T.Longitude<=max(S.lon)&T.Latitude>=min(S.lat)&T.Latitude<=max(S.lat);
T=T(keep,:); writetable(T,fullfile(caseDir,'Tibet_Modern_samples.xlsx'));
runText=fileread(fullfile(anc,'Tibet_Eocene_30Ma_SmoothGrid_G4000_Q3500_V4000_Best.run'));
runText=strrep(runText,'Tibet_Eocene_30Ma','Tibet_Modern'); runText=strrep(runText,'Tibet_Modern_samples.xlsx','Tibet_Modern_samples.xlsx');
lines=splitlines(string(runText));
lines(4)="Modern Tibet OPI reference using present-day topography and isotope observations";
% ensure case-relative data file names
lines(6)=string(caseDir); lines(8)="Tibet_Modern_topo.mat"; lines(10)="Tibet_Modern_samples.xlsx"; lines(11)="Tibet_Modern_topo_divide_main.mat";
fid=fopen(fullfile(caseDir,'Tibet_Modern_Best.run'),'w'); fprintf(fid,'%s\n',lines); fclose(fid);
fprintf('Prepared modern case: %s\nSamples: %d\n',caseDir,height(T));
end
