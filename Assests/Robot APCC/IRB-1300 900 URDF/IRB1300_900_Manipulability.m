clear;
%Load urdf file
Robot=importrobot('urdf/IRB-1300 900 URDF.urdf');
Robot.Gravity = [0 0 -9.81];
%Generate workspace and configurations
[wksp,cfgs] = generateRobotWorkspace(Robot,{},MaxNumSamples=1000);
wsAlpha = alphaShape(wksp(:,1),wksp(:,2),wksp(:,3));



%Generate manipulability index
% MIminTrim=0.08;
% MI_InvCond = manipulabilityIndex(Robot,cfgs,IndexType ="inverse-condition");
% wksp_trimmed = wksp(MI_InvCond>MIminTrim,:);
% MI_InvCond_trimmed = MI_InvCond(MI_InvCond>MIminTrim,:);

MIminTrim=0.1;
MI_Yoshikawa = manipulabilityIndex(Robot,cfgs,IndexType="yoshikawa");
wksp_trimmed = wksp(MI_Yoshikawa>MIminTrim,:);
MI_Yoshikawa_trimmed = MI_Yoshikawa(MI_Yoshikawa>MIminTrim,:);

%Define Table
Table= collisionBox(.822,1.137,.15);
T = trvec2tform([.2475 0 -.092]);
Table.Pose = T;


%Generate plots
show(Robot,homeConfiguration(Robot),Visuals="on");
hold on
a = plot(wsAlpha,FaceAlpha=0.45,EdgeColor="none");
% showWorkspaceAnalysis(wksp_trimmed,MI_InvCond_trimmed,Voxelize=false);
%showWorkspaceAnalysis(wksp,MI_InvCond,Voxelize=false);
showWorkspaceAnalysis(wksp_trimmed,MI_Yoshikawa_trimmed,Voxelize=false);
[~, patchObj] = show(Table);
patchObj.FaceColor = [0 1 1];


title("Manipulability Index - Inverse Condition - IRB1300-900");
%title("Manipulability Index - Yoshikawa - IRB1300-900");
%view(0,90)
axis auto

hold off

%OutputGH=[wksp, IndexInvCond];
% OutputGH=[wksp_trimmed, MI_InvCond_trimmed];
%OutputGH=[wksp, MI_Yoshikawa];
OutputGH=[wksp_trimmed, MI_Yoshikawa_trimmed];
writematrix(OutputGH,'IRB_1300-900_Robot_MI_Yoshikawa_100000points_Trimmed0.1.csv');
% writematrix(OutputGH,'IRB_1300-900_Robot_InvCond_100000points_Trimmed0.08.csv');