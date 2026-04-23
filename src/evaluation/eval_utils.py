import sys
sys.path.append("..")
sys.path.append(".")
from typing import DefaultDict
import pickle
SAVE_LOG = False

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from env.vehicle import Status
from env.map_level import get_map_level
from configs import *

def eval(env, agent, episode=2000, log_path='', multi_level=False, post_proc_action=True):

    succ_rate_case = DefaultDict(list)
    if multi_level:
        succ_rate_level = DefaultDict(list)
        step_num_level = DefaultDict(list)
        path_length_level = DefaultDict(list)
    reward_case = DefaultDict(list)
    reward_record = []
    succ_record = []
    success_step_record = []
    step_record = DefaultDict(list)
    path_length_record = DefaultDict(list)
    eval_record = []

    for i in trange(episode):
        obs = env.reset(i+1)
        agent.reset()
        prev_action = None  # 相对动作掩码
        done = False
        total_reward = 0
        step_num = 0
        path_length = 0
        last_xy = (env.vehicle.state.loc.x, env.vehicle.state.loc.y)
        last_obs = obs['target']
        episode_steers = []  # 平顺度：记录本episode所有steer
        while not done:
            step_num += 1
            if post_proc_action:
                action, _ = agent.choose_action(obs, prev_action)
            else:
                action, _ = agent.get_action(obs, prev_action)
            episode_steers.append(action[0])  # 平顺度：记录steer
            if (last_obs == obs['target']).all():
                action = env.action_space.sample()
            last_obs = obs['target']
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            obs = next_obs
            prev_action = action  # 相对动作掩码
            path_length += np.linalg.norm(np.array(last_xy)-np.array((env.vehicle.state.loc.x, env.vehicle.state.loc.y)))
            last_xy = (env.vehicle.state.loc.x, env.vehicle.state.loc.y)
            
            if info['path_to_dest'] is not None:
                agent.set_planner_path(info['path_to_dest'])
            if done:
                if info['status']==Status.ARRIVED:
                    succ_record.append(1)
                else:
                    succ_record.append(0)

        reward_record.append(total_reward)
        succ_rate_case[env.map.case_id].append(succ_record[-1])
        if step_num < 200:
            path_length_record[env.map.case_id].append(path_length)
        reward_case[env.map.case_id].append(reward_record[-1])
        if multi_level:
            succ_rate_level[env.map.map_level].append(succ_record[-1])
            if step_num < 200:
                path_length_level[env.map.map_level].append(path_length)
            step_num_level[env.map.map_level].append(step_num)
        if info['status']==Status.OUTBOUND:
            step_record[env.map.case_id].append(200)
        else:
            step_record[env.map.case_id].append(step_num)
        if succ_record[-1] == 1:
            success_step_record.append(step_num)
        eval_record.append({'case_id':env.map.case_id,
                            'status':info['status'],
                            'step_num':step_num,
                            'reward':total_reward,
                            'path_length':path_length,
                            'steer_sequence': episode_steers.copy(),
                            })

    # 平顺度：计算所有episodes的|Δsteer|均值
    all_delta_steer = []
    for r in eval_record:
        steers = r['steer_sequence']
        for j in range(1, len(steers)):
            all_delta_steer.append(abs(steers[j] - steers[j-1]))
    smoothness = np.mean(all_delta_steer) if all_delta_steer else 0
    # 效率：E = 1e-7 * (SR/100) / T
    avg_step = np.mean([r['step_num'] for r in eval_record])
    sr = np.mean(succ_record)
    efficiency = 1e-7 * (sr / 100) / avg_step if avg_step > 0 else 0

    print('#'*15)
    print('EVALUATE RESULT:')
    print('success rate: ', np.mean(succ_record))
    print('average reward: ', np.mean(reward_record))
    print(f'smoothness (mean|Δsteer|): {smoothness:.6f}')
    print(f'efficiency: {efficiency:.6e}')
    print('-'*10)
    print('success rate per case: ')
    case_ids = [int(k) for k in succ_rate_case.keys()]
    case_ids.sort()
    if len(case_ids) < 10:
        print('-'*10)
        print('average reward per case: ')
        for k in case_ids:
            env.reset(k)
            print('case %s (%s) :'%(k,get_map_level(env.map.start, env.map.dest, env.map.obstacles))\
                , np.mean(succ_rate_case[k]))
        for k in case_ids:
            print('case %s :'%k, np.mean(reward_case[k]), np.mean(step_record[k]), '+-(%s)'%np.std(step_record[k]))

    if multi_level:
        print('success rate per level: ')
        for k in succ_rate_level.keys():
            print('%s (case num %s):'%(k, len(succ_rate_level[k])) + '%s '%np.mean(succ_rate_level[k]))
    
    if log_path is not None:
        def plot_time_ratio(node_list):
            max_node = TOLERANT_TIME
            raw_len = len(node_list)
            filtered_node_list = []
            for n in node_list:
                if n != max_node:
                    filtered_node_list.append(n)
            filtered_node_list.sort()
            ratio_list = [i/raw_len for i in range(1,len(filtered_node_list)+1)]
            plt.plot(filtered_node_list, ratio_list)
            plt.xlabel('Search node')
            plt.ylabel('Accumulate success rate')
            fig = plt.gcf()
            fig.savefig(log_path+'/success_rate.png')
            plt.close()
        all_step_record = []
        for k in step_record.keys():
            all_step_record.extend(step_record[k])
        plot_time_ratio(all_step_record)

        # save eval result
        f_record = open(log_path+'/record.data', 'wb')
        pickle.dump(eval_record, f_record)
        f_record.close()

        f_record_txt = open(log_path+'/result.txt', 'w', newline='')
        f_record_txt.write('success rate: %s\n'%np.mean(succ_record))
        f_record_txt.write('average reward: %s\n'%np.mean(reward_record))
        f_record_txt.write(f'smoothness (mean|Δsteer|): {smoothness:.6f}\n')
        f_record_txt.write(f'efficiency: {efficiency:.6e}\n')
        f_record_txt.write('step num: %s '%np.mean(success_step_record)+'+-(%s)\n'%np.std(success_step_record))
        if multi_level:
            f_record_txt.write('\n')
            for k in succ_rate_level.keys():
                f_record_txt.write('%s (case num %s):'%(k, len(succ_rate_level[k])) + '%s \n'%np.mean(succ_rate_level[k]))
                f_record_txt.write('step num: %s '%np.mean(step_num_level[k])+'+-(%s)\n'%np.std(step_num_level[k]))
                f_record_txt.write('path length: %s '%np.mean(path_length_level[k])+'+-(%s)\n'%np.std(path_length_level[k]))
        if len(case_ids) < 10:
            for k in case_ids:
                f_record_txt.write('\ncase %s : '%k + 'success rate: %s \n'%np.mean(succ_rate_case[k]))
                f_record_txt.write('step num: %s '%np.mean(step_record[k])+'+-(%s)\n'%np.std(step_record[k]))
                f_record_txt.write('path length: %s '%np.mean(path_length_record[k])+'+-(%s)\n'%np.std(path_length_record[k]))
        f_record_txt.close()
    
    return np.mean(succ_record)
