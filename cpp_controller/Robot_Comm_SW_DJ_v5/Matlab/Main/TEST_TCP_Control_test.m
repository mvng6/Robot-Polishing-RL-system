clear;
close all;
clc;

% Initialization;
% Figure_setup;

%%
fname = 'Data_Speedl_F_35_P_80.00_I_130.00_D_0.50.csv';
data = readmatrix(fname);
Target_F = sscanf(fname, 'Data_Speedl_F_%d_P_%*f_I_%*f_D_%*f.txt', 1);

t = (data(:,1) - data(1,1))*0.001;
flange_pos = data(:,2:7);
tcp_pos = data(:,8:13);
joint = data(:,14:19);
tcp_vel = data(:,20:25);
joint_vel = data(:,26:31);

%%
figure;
for i = 1:6
    subplot(2,3,i);
    plot(t,tcp_pos(:,i));
    xlim([0 t(end)])
end

figure;
for i = 1:6
    subplot(2,3,i);
    plot(t,tcp_vel(:,i));
    xlim([0 t(end)])
end