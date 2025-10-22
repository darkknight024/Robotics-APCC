clear;
% Load urdf file
Robot=importrobot('urdf/IRB-1300 900 URDF.urdf');
Robot.Gravity = [0 0 -9.81];

% Generate workspace and configurations
[wksp,cfgs] = generateRobotWorkspace(Robot,{},MaxNumSamples=800000);
%wsAlpha = alphaShape(wksp(:,1),wksp(:,2),wksp(:,3));

% Generate manipulability index
%____Yoshikawa Index___%
MI_Yoshikawa = manipulabilityIndex(Robot,cfgs,IndexType="yoshikawa");
MaxMI=max(MI_Yoshikawa);
Top10Trim=0.9*MaxMI
wksp_trimmed = wksp(MI_Yoshikawa>Top10Trim,:);
MI_Yoshikawa_trimmed = MI_Yoshikawa(MI_Yoshikawa>Top10Trim,:);
% Define Table
Table= collisionBox(.822,1.137,.15);
T = trvec2tform([.2475 0 -.092]);
Table.Pose = T;
[~, patchObj] = show(Table);
patchObj.FaceColor = [0 1 1];
% Generate plot
show(Robot,homeConfiguration(Robot),Visuals="on");
hold on
showWorkspaceAnalysis(wksp_trimmed,MI_Yoshikawa_trimmed,Voxelize=false);
title("Manipulability Index - Yoshikawa - IRB1300-900");
axis auto
hold off
%Save output file
OutputGH=[wksp, MI_Yoshikawa];
writematrix(OutputGH,'IRB_1300-900_Robot_MI_Yoshikawa_800000points.csv');
%Save output file trimmed
OutputGHtrimmed=[wksp_trimmed, MI_Yoshikawa_trimmed];
writematrix(OutputGHtrimmed,'IRB_1300-900_Robot_MI_Yoshikawa_800000points_Top10Percent.csv');


% %____Inverse-Condition Number___%
% MI_InvCond = manipulabilityIndex(Robot,cfgs,IndexType ="inverse-condition");
% MaxMI=max(MI_InvCond);
% Top10Trim=0.9*MaxMI
% wksp_trimmed = wksp(MI_InvCond>Top10Trim,:);
% MI_InvCond_trimmed = MI_InvCond(MI_InvCond>Top10Trim,:);
% % Define Table
% Table= collisionBox(.822,1.137,.15);
% T = trvec2tform([.2475 0 -.092]);
% Table.Pose = T;
% [~, patchObj] = show(Table);
% patchObj.FaceColor = [0 1 1];
% % Generate plot
% show(Robot,homeConfiguration(Robot),Visuals="on");
% hold on
% showWorkspaceAnalysis(wksp_trimmed,MI_InvCond_trimmed,Voxelize=false);
% %showWorkspaceAnalysis(wksp,MI_InvCond,Voxelize=false);
% title("Manipulability Index - Inverse Condition - IRB1300-900");
% axis auto
% hold off
% %OutputGH=[wksp, IndexInvCond];
% OutputGH=[wksp_trimmed, MI_InvCond_trimmed];
% %writematrix(OutputGH,'IRB_1300-900_Robot_InvCond_100000points_Trimmed0.08.csv');