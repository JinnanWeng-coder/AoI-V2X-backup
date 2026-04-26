import time
import numpy as np
import os
import scipy.io
import Classes.Environment_Platoon as ENV
import random as _random


SEED = 2
_random.seed(SEED)
np.random.seed(SEED)

start = time.time()

# ################## SETTINGS ######################
up_lanes = [i / 2.0 for i in
            [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]]
down_lanes = [i / 2.0 for i in
              [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2,
               750 - 3.5 / 2]]
left_lanes = [i / 2.0 for i in
              [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]]
right_lanes = [i / 2.0 for i in
               [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2,
                1299 - 3.5 / 2]]
print('------------- lanes are -------------')
print('up_lanes :', up_lanes)
print('down_lanes :', down_lanes)
print('left_lanes :', left_lanes)
print('right_lanes :', right_lanes)
print('------------------------------------')

width = 750 / 2
height = 1298 / 2
label = 'marl_model'

# ------------------------------------------------------------------------------------------------------------------ #
# simulation parameters:
# ------------------------------------------------------------------------------------------------------------------ #
size_platoon = 4
n_veh = 20
n_platoon = int(n_veh / size_platoon)
n_RB = 3
n_S = 2
Gap = 25
max_power = 30
V2I_min = 540
bandwidth = int(180000)
V2V_size = int((4000) * 8)

env = ENV.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, size_platoon, n_RB,
                  V2I_min, bandwidth, V2V_size, Gap)
env.new_random_game()

n_episode = 500
n_step_per_episode = int(env.time_slow / env.time_fast)
n_episode_test = 100


def get_state(env, idx):
    """Get state from the environment, kept for parity with Main.py."""
    V2I_abs = (env.V2I_channels_abs[idx * size_platoon] - 60) / 60.0
    V2V_abs = (env.V2V_channels_abs[idx * size_platoon, idx * size_platoon + (1 + np.arange(size_platoon - 1))] - 60) / 60.0
    V2I_fast = (env.V2I_channels_with_fastfading[idx * size_platoon, :] - env.V2I_channels_abs[
        idx * size_platoon] + 10) / 35
    V2V_fast = (env.V2V_channels_with_fastfading[idx * size_platoon, idx * size_platoon + (1 + np.arange(size_platoon - 1)), :]
                - env.V2V_channels_abs[idx * size_platoon, idx * size_platoon +
                                       (1 + np.arange(size_platoon - 1))].reshape(size_platoon - 1, 1) + 10) / 35
    Interference = (env.Interference_all[idx] + 60) / 60
    AoI_levels = env.AoI[idx] / (int(env.time_slow / env.time_fast))
    V2V_load_remaining = np.asarray([env.V2V_demand[idx] / env.V2V_demand_size])
    return np.concatenate((np.reshape(V2I_abs, -1), np.reshape(V2I_fast, -1), np.reshape(V2V_abs, -1),
                           np.reshape(V2V_fast, -1), np.reshape(Interference, -1), np.reshape(AoI_levels, -1),
                           V2V_load_remaining), axis=0)


n_input = len(get_state(env=env, idx=0))
n_output = 3

AoI_evolution = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)
Demand_total = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)
V2I_total = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)
V2V_total = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)
power_total = np.zeros([n_platoon, n_episode_test, n_step_per_episode], dtype=np.float16)

AoI_total = np.zeros([n_platoon, n_episode], dtype=np.float16)
record_reward_t1_ = np.zeros([n_platoon, n_episode], dtype=np.float16)
record_reward_t2_ = np.zeros([n_platoon, n_episode], dtype=np.float16)
record_reward_global_ = np.zeros([n_episode], dtype=np.float16)

# Target: RB=1, mode=0, power=15.
fixed_action = np.array([-1.0 / 3.0, -0.999, 0.0])

for i_episode in range(n_episode):
    if i_episode % 20 == 0:
        print('Episode:', i_episode)

    record_reward_t1 = np.zeros([n_platoon, n_step_per_episode], dtype=np.float16)
    record_reward_t2 = np.zeros([n_platoon, n_step_per_episode], dtype=np.float16)
    record_reward_global = np.zeros([n_step_per_episode], dtype=np.float16)
    record_AoI = np.zeros([n_platoon, n_step_per_episode], dtype=np.float16)

    env.V2V_demand = env.V2V_demand_size * np.ones(n_platoon, dtype=np.float16)
    env.individual_time_limit = env.time_slow * np.ones(n_platoon, dtype=np.float16)
    env.active_links = np.ones((int(env.n_Veh / env.size_platoon)), dtype='bool')
    if i_episode == 0:
        env.AoI = np.ones(int(n_platoon)) * 100

    if i_episode % 20 == 0:
        env.renew_positions()
        env.renew_channel(n_veh, size_platoon)
        env.renew_channels_fastfading()

    for i_step in range(n_step_per_episode):
        action_all = []
        action_all_training = np.zeros([n_platoon, n_output], dtype=int)

        for i in range(n_platoon):
            action = fixed_action.copy()
            action_all.append(action)
            action_all_training[i, 0] = ((action[0] + 1) / 2) * n_RB
            action_all_training[i, 1] = ((action[1] + 1) / 2) * n_S
            action_all_training[i, 2] = np.round(np.clip(((action[2] + 1) / 2) * max_power, 1, max_power))

        action_temp = action_all_training.copy()
        task_1_r, task_2_r, global_reward, platoon_AoI, C_rate, V_rate, Demand_R, V2V_success = \
            env.act_for_training(action_temp)

        for i in range(n_platoon):
            record_reward_t1[i, i_step] = task_1_r[i]
            record_reward_t2[i, i_step] = task_2_r[i]
            record_AoI[i, i_step] = env.AoI[i]
        record_reward_global[i_step] = global_reward

        env.renew_channels_fastfading()
        env.Compute_Interference(action_temp)

        for i in range(n_platoon):
            AoI_evolution[i, i_episode % n_episode_test, i_step] = platoon_AoI[i]
            Demand_total[i, i_episode % n_episode_test, i_step] = Demand_R[i]
            V2I_total[i, i_episode % n_episode_test, i_step] = C_rate[i]
            V2V_total[i, i_episode % n_episode_test, i_step] = V_rate[i]
            power_total[i, i_episode % n_episode_test, i_step] = action_temp[i, 2]

    record_reward_t1_[:, i_episode] = np.mean(record_reward_t1, axis=1)
    record_reward_t2_[:, i_episode] = np.mean(record_reward_t2, axis=1)
    record_reward_global_[i_episode] = np.mean(record_reward_global)
    AoI_total[:, i_episode] = np.mean(record_AoI, axis=1)

print('Baseline fixed done. Saving results...')
current_dir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(current_dir, 'model', label, 'baseline_fixed_seed1')
os.makedirs(save_dir, exist_ok=True)

scipy.io.savemat(os.path.join(save_dir, 'reward_t1.mat'), {'reward_t1': record_reward_t1_})
scipy.io.savemat(os.path.join(save_dir, 'reward_t2.mat'), {'reward_t2': record_reward_t2_})
scipy.io.savemat(os.path.join(save_dir, 'reward_global.mat'), {'reward_global': record_reward_global_})
scipy.io.savemat(os.path.join(save_dir, 'AoI.mat'), {'AoI': AoI_total})
scipy.io.savemat(os.path.join(save_dir, 'AoI_evolution.mat'), {'AoI_evolution': AoI_evolution})
scipy.io.savemat(os.path.join(save_dir, 'demand.mat'), {'demand': Demand_total})
scipy.io.savemat(os.path.join(save_dir, 'V2I.mat'), {'V2I': V2I_total})
scipy.io.savemat(os.path.join(save_dir, 'V2V.mat'), {'V2V': V2V_total})
scipy.io.savemat(os.path.join(save_dir, 'power.mat'), {'power': power_total})

end = time.time()
print("simulation took this much time ... ", end - start)
