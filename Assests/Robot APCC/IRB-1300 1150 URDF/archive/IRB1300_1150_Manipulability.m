clear;
%Load urdf file
Robot=importrobot('urdf/IRB 1300-1150 URDF.urdf');
Robot.Gravity = [0 0 -9.81];
%Generate workspace and configurations
[wksp,cfgs] = generateRobotWorkspace(Robot,{},MaxNumSamples=1000);


MIminTrim=0.08;

%Generate manipulability index
MI_InvCond = manipulabilityIndex(Robot,cfgs,IndexType ="inverse-condition");
wksp_trimmed = wksp(MI_InvCond>MIminTrim,:);
MI_InvCond_trimmed = MI_InvCond(MI_InvCond>MIminTrim,:);

% MI_Yoshikawa = manipulabilityIndex(Robot,cfgs,IndexType="yoshikawa");
% wksp_trimmed = wksp(MI_Yoshikawa>MIminTrim,:);
% MI_Yoshikawa_trimmed = MI_Yoshikawa(MI_Yoshikawa>MIminTrim,:);

%Define Table
Table= collisionBox(.822,1.137,.15);
T = trvec2tform([.2475 0 -.092]);
Table.Pose = T;


%Generate plots
show(Robot,homeConfiguration(Robot),Visuals="on");
hold on

showWorkspaceAnalysis(wksp_trimmed,MI_InvCond_trimmed,Voxelize=false);
%showWorkspaceAnalysis(wksp,MI_InvCond,Voxelize=false);
%showWorkspaceAnalysis(wksp_trimmed,MI_Yoshikawa_trimmed,Voxelize=false);
[~, patchObj] = show(Table);
patchObj.FaceColor = [0 1 1];


title("Manipulability Index - Inverse Condition - IRB1300-1150");
%title("Manipulability Index - Yoshikawa - IRB1300-1150");
%view(0,90)
axis auto

hold off

%OutputGH=[wksp, IndexInvCond];
OutputGH=[wksp_trimmed, MI_InvCond_trimmed];
%OutputGH=[wksp, MI_Yoshikawa];
%writematrix(OutputGH,'IRB_1300-1150_Robot_MI_Yoshikawa_15000points.csv');
writematrix(OutputGH,'IRB_1300-1150_Robot_InvCond_100000points_Trimmed0.08.csv');