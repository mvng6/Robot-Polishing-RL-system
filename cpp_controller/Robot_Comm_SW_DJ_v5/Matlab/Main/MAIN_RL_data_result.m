clear;
close all;
clc;

%%
data = readmatrix('Data_Speedl_F_35_P_80.00_I_130.00_D_0.50.csv');

t = (data(:,1) - data(1,1)) * 0.001;

RL_p = data(:,end-2);
RL_timing = data(:,end-1);
RL_ep_flag = data(:,end);

%%
RL_ep_flag_sum = sum(RL_ep_flag)

%%
figure;
subplot(3,1,1);
plot(t,RL_p);
xlim([0 t(end)]);

subplot(3,1,2);
plot(t,RL_timing);
xlim([0 t(end)]);

subplot(3,1,3);
plot(t,RL_ep_flag);
xlim([0 t(end)]);