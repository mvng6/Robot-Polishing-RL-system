clear;
close all;
clc;

%% C++ 기록 데이터
data = readmatrix("Data_Speedl_F_30_P_80.00_I_130.00_D_0.50.csv");

t = (data(:,1) - data(1,1)) * 0.001;

RL_u = data(:,end-2);
RL_send_flag = data(:,end-1);
RL_ep_flag = data(:,end);

%% Python -> C++ 보내기 직전 데이터
pre_before_data = readtable('packet_intended_20250901_175024_ep02.csv');

before_data = table2cell(pre_before_data);
before_rl_data = cell2mat(before_data(:,5));
before_msg_flag = zeros(length(before_data),1);
before_ep_flag = zeros(length(before_data),1);

for i = 1:length(before_msg_flag)
    if strcmp(before_data{i,6},'True')
        before_msg_flag(i) = 1;
    else
        before_msg_flag(i) = 0;
    end
end

for i = 1:length(before_ep_flag)
    if strcmp(before_data{i,7},'True')
        before_ep_flag(i) = 1;
    else
        before_ep_flag(i) = 0;
    end
end

%% Python -> C++ 보내고 난 후 데이터
pre_after_data = readtable('packet_sent_20250901_175024_ep02.csv');

after_data = table2cell(pre_after_data);
after_rl_data = cell2mat(after_data(:,7));
after_msg_flag = zeros(length(before_data),1);
after_ep_flag = zeros(length(after_data),1);

for i = 1:length(after_msg_flag)
    if strcmp(after_data{i,8},'True')
        after_msg_flag(i) = 1;
    else
        after_msg_flag(i) = 0;
    end
end

for i = 1:length(after_ep_flag)
    if strcmp(after_data{i,9},'True')
        after_ep_flag(i) = 1;
    else
        after_ep_flag(i) = 0;
    end
end

%%
figure;
subplot(3,1,1);
plot(RL_u);
hold on;
% plot(before_rl_data,'r--');
% plot(after_rl_data,'b-.');
% legend('C++','Before send','After send','Location','northeastoutside');
% legend('Before send','After send','Location','northeastoutside');

subplot(3,1,2);
plot(RL_send_flag);
hold on;
% plot(before_msg_flag,'r--');
% plot(after_msg_flag,'b-.');
% legend('C++','Before send','After send','Location','northeastoutside');
% legend('Before send','After send','Location','northeastoutside');

subplot(3,1,3);
plot(RL_ep_flag);
hold on;
% plot(before_ep_flag,'r--');
% plot(after_ep_flag,'b-.');
% legend('C++','Before send','After send','Location','northeastoutside');
% legend('Before send','After send','Location','northeastoutside');


%%
% figure;
% subplot(3,1,1);
% plot(t,RL_u);
% xlim([0 t(end)]);
% ylabel('RL u')
% subplot(3,1,2);
% plot(t,RL_send_flag);
% xlim([0 t(end)]);
% ylabel('Send Message Flag');
% subplot(3,1,3);
% plot(t,RL_ep_flag);
% xlim([0 t(end)]);
% ylabel('Episode Flag');