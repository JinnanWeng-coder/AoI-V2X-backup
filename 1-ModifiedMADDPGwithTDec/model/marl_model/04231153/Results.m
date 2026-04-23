%% Algorithm 1 (Modified MADDPG with TDec) 结果可视化脚本
clear; clc; close all;

% 设置绘图风格
set(0, 'DefaultLineLineWidth', 1.5);
set(0, 'DefaultAxesFontSize', 12);
set(0, 'DefaultAxesFontWeight', 'bold');
set(0, 'DefaultFigureColor', [1 1 1]);

%% 1. 任务分解奖励对比 (Task-specific Rewards)
if exist('reward_t1.mat', 'file') && exist('reward_t2.mat', 'file')
    load('reward_t1.mat'); 
    load('reward_t2.mat'); 
    
    avg_t1 = mean(reward_t1, 1);
    avg_t2 = mean(reward_t2, 1);
    
    figure('Name', 'Task Decomposition Rewards');
    set(gcf, 'Color', 'w'); % 设置白色背景
    
    % 使用 movmean 平滑处理，并增加线宽
    plot(avg_t1, 'b', 'LineWidth', 1.5, 'DisplayName', 'Task 1: V2V Demand');
    hold on;
    plot(avg_t2, 'r', 'LineWidth', 1.5, 'DisplayName', 'Task 2: V2I & AoI');
    
    grid on;
    xlabel('Episode');
    ylabel('Local Reward');
    title('Convergence of Decomposed Tasks');
    
    % --- 核心修改：图例移至下方外部，水平排列 ---
    legend('Location', 'southoutside', 'Orientation', 'horizontal');
    % -----------------------------------------
    
    hold off; 
    saveas(gcf, 'Algo1_Result_1_Task_Rewards.png');
end

%% 2. 全局奖励曲线 (Global Reward)
if exist('reward_global.mat', 'file')
    load('reward_global.mat');
    if size(reward_global, 1) > 1
        avg_global = mean(reward_global, 1);
    else
        avg_global = reward_global;
    end
    
    figure('Name', 'Global Reward');
    plot(movmean(avg_global, 20), 'Color', [0.4660 0.6740 0.1880]);
    grid on;
    title('Global System Performance (Interference Mitigation)');
    xlabel('Episode');
    ylabel('Global Reward');
    saveas(gcf, 'Algo1_Result_2_Global_Reward.png');
end

%% 3. AoI 累积分布函数 (CDF - 替换平均趋势图)
if exist('AoI.mat', 'file')
    load('AoI.mat'); % 假设变量名为 AoI [n_platoon, n_episode]
    
    % 为了体现收敛后的性能，我们取最后 100 个 Episode 的所有车队数据
    % 如果总 Episode 较少，可以根据实际情况调整，例如 AoI(:, end-50:end)
    if size(AoI, 2) >= 100
        stable_aoi_data = AoI(:, end-99:end);
    else
        stable_aoi_data = AoI;
    end
    
    % 将矩阵拉平为一维向量，用于计算分布
    aoi_vector = stable_aoi_data(:);
    
    % 计算经验累积分布函数 (Empirical CDF)
    [f, x] = ecdf(aoi_vector);
    
    figure('Name', 'AoI CDF');
    % 绘制曲线，设置粗细和颜色
    plot(x, f, 'LineWidth', 2, 'Color', [0 0.4470 0.7410]);
    grid on;
    
    % 装饰图表
    xlabel('Age of Information (AoI)');
    ylabel('Probability (P \leq x)');
    title('CDF of Average AoI (Last 100 Episodes)');
    
    % 设置坐标轴范围，方便观察（根据你数据的最大值微调）
    xlim([0 max(x)*1.1]);
    ylim([0 1.05]);
    
    % 添加一些辅助说明，比如 90% 的 AoI 都在多少以下
    p90 = prctile(aoi_vector, 90);
    line([p90 p90], [0 0.9], 'Color', 'r', 'LineStyle', '--');
    text(p90, 0.4, ['  90th Percentile: ', num2str(round(p90,2))], 'Color', 'r');
    
    saveas(gcf, 'Algo1_Result_3_AoI_CDF.png');
end

%% 4. 实时 AoI 演进 (Intra-Episode Evolution - 新增)
if exist('AoI_evolution.mat', 'file')
    load('AoI_evolution.mat'); % 变量名为 AoI_evolution
    % 结构通常为 [n_platoon, 100, n_step]
    % 提取最后 100 轮记录中最后一轮的平均值
    last_episode_aoi = squeeze(mean(AoI_evolution(:, end, :), 1));
    
    figure('Name', 'AoI Real-time Evolution');
    plot(last_episode_aoi, 'Color', [0.3010 0.7450 0.9330], 'LineWidth', 2);
    grid on;
    title('Intra-Episode: AoI Sawtooth Evolution (Algorithm 1)');
    xlabel('Step');
    ylabel('Instantaneous AoI');
    % 设置坐标轴以更好地观察锯齿
    ylim([0 max(last_episode_aoi) + 5]);
    saveas(gcf, 'Algo1_Result_4_AoI_Evolution.png');
end

%% 5. V2V 需求消耗 (Intra-Episode)
if exist('demand.mat', 'file')
    load('demand.mat');
    last_demand = squeeze(mean(demand(:, end, :), 1));
    
    figure('Name', 'V2V Demand Depletion');
    plot(last_demand, 'm', 'LineWidth', 2);
    grid on;
    title('Intra-Episode: V2V Demand Depletion');
    xlabel('Step');
    ylabel('Remaining Demand (bits)');
    saveas(gcf, 'Algo1_Result_5_Demand_Step.png');
end

%% 6. V2I 与 V2V 速率对比
if exist('V2I.mat', 'file') && exist('V2V.mat', 'file')
    load('V2I.mat');
    load('V2V.mat');
    mean_v2i = squeeze(mean(mean(V2I, 1), 2));
    mean_v2v = squeeze(mean(mean(V2V, 1), 2));
    
    figure('Name', 'Rate Comparison');
    plot(mean_v2i, 'r', 'DisplayName', 'V2I Rate (Cellular)');
    hold on;
    plot(mean_v2v, 'k', 'DisplayName', 'V2V Rate (Vehicle)');
    grid on;
    legend show;
    title('Algorithm 1: Multi-Task Transmission Rate');
    xlabel('Step');
    ylabel('Rate (bps)');
    saveas(gcf, 'Algo1_Result_6_Rate_Comparison.png');
end
%% 7. 平均 AoI 趋势 (Episode Level)
if exist('AoI.mat', 'file')
    load('AoI.mat');
    avg_aoi = mean(AoI, 1);
    
    figure('Name', 'AoI Trend');
    plot(avg_aoi, 'Color', [0 0.4470 0.7410]);
    grid on;
    title('Algorithm 1: Average AoI per Episode');
    xlabel('Episode');
    ylabel('Average AoI');
    saveas(gcf, 'Algo1_Result_3_AoI_Trend.png');
end