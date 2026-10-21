import gymnasium as gym
from gymnasium import spaces
from numpy import *
import json
import random
import torch
import torch.nn as nn
import pandas as pd

# Action 2 dimension 3 inventory levels with num_station combinations
# ds: the current inventory of stations
# S: the number of stations
# V: the number of vehicles
# Cap_s : the capacity of stations
# Cap_v : the capacity of vehicles
# NS: the neighborhood of each station
# dv: the current inventory of vehicles
# loc_v : the current location of vehicles
# Dis: distance of each pair of stations
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(array(env.single_observation_space.shape).prod(), 1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            #nn.Linear(512,256),
            #nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )
    def forward(self, x):
        return self.network(x)

class BssRouteEM(gym.Env):
    ###########################################################################
    ###Define Environment
    ###########################################################################
    def __init__(self):
        self.num_station = 60
        self.V = 4
        self.cap_s = [40] * 9 + [20] * 51
        self.opert = 1.5  # operation time in min
        with open('/local_workspace/lianjiaq/Instances/Ins_Pro2/dis_gen60.json') as fp:
            dis = json.load(fp)
        dis = array(dis)
        self.Tran_tim = 2 * dis
        with open('/local_workspace/lianjiaq/Instances/Ins_Pro2/NS_gen60.json') as fp:
            NS = json.load(fp)
        self.NS = array(NS)
        with open('/local_workspace/lianjiaq/Instances/Ins_Pro2/base2t.json') as fp:
            self.ini_ds = json.load(fp)  # initial inventory
        self.cap_V = [40] * self.V
        self.start_time = 420
        self.end_time = 660  # in minutes
        self.observation_space = spaces.Box(
            low=hstack((zeros((self.V * self.num_station)), zeros((self.V * self.num_station)), zeros((self.V)),
                        zeros((self.V)), zeros((self.num_station)), 0)),
            high=hstack((ones((self.V * self.num_station)), ones((self.V * self.num_station)), ones((self.V)),
                         ones((self.V)), ones((self.num_station)), 1)), dtype=float32)
        # {'Vehicle_location': current station one hot vector 60x4
        #  'Vehicle_target': target station one hot vector 60x4
        #  'Vehicle_inventory': current inventory for each vehicle 4 dimension array 0-capacity
        #  'Vehicle_eta': eta in mins 4 dimension array start-time - end of day
        #  'Station': inventory of stations 60 dimension array 0-capacity
        #  'time_stamp': single value
        # example [one hot 60 for v1, one hot 60 for v2, one hot 60 for v3, one hot 60 for v4,
        #          one hot 60 for v1, one hot 60 for v2, one hot 60 for v3, one hot 60 for v4,
        #          inventory_v1, inventory_v2, inventory_v3, inventory_v4, eta_v1, eta_v2, eta_v3, eta_v4,
        #          inventory_station1 - 60, current time in min]
        #self_station_level = 3
        # self.neighbour = self.num_station - self.V
        self.action_space = spaces.Discrete(self.num_station) # three levels to rebalance the stations with stations
        env_id ='BssRebalT-v0'
        capture_video = True
        run_name = 'BssRebalT-v0__DQN_datasetE__1__1711475942'
        self.device = torch.device("cpu")
        model_path = 'runs/BssRebalT-v0__DQN_datasetE__1__1786387106/DQN_datasetE.cleanrl_model'
        #model_path = 'runs/BssRebalT-v0__DQN_datasetE__1__1715969404/DQN_datasetE.cleanrl_model'
        #envs = gym.vector.SyncVectorEnv([self.make_env(env_id, 0, capture_video, run_name)])
        self.model = QNetwork(envs).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.index_tar = []
        for j in range(5):  #
            self.index_tar.append('period' + str(j + 1) + 'Target')

    def make_env(self, env_id, seed, capture_video, run_name):
        def thunk():
            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            return env

        return thunk
    ##########################################################################
    # Construct Observations from environment states
    ##########################################################################
    def _get_obs(self):
        self.normaldv = array(self.dv) / self.cap_V
        self.normal_est_tartime = (array(self.est_tartime) - self.start_time) / 250
        self.normalds = array(self.ds) / array(self.cap_s)
        self.tmstamp1 = (self.tmstamp - self.start_time) / 250
        obs = hstack(
            (self.loc_v, self.tar_loc, self.normaldv, self.normal_est_tartime, self.normalds, self.tmstamp1))
        return obs.astype(float32)

    def _get_info(self):
        return {'Trips': self.demand_request, 'Ongoing_oper': self.w, 'Ongoing_reb': self.ongoing_reb, 'neigh': self.neighset, 'tar': self.tar_all}

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
        self.ds = self.ini_ds
        # i = random.randint(0, 149)
        i = random.choice([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                           67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
                           84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99])
        print('Day', i + 1)
        with open('/local_workspace/lianjiaq/Instances/Ins_Pro2/trip' + str(i + 350) + '.json') as fp:
            self.demand_request = json.load(fp)
        df_inter = pd.read_csv('/local_workspace/lianjiaq/Instances/Ins_Pro2/Instance2/InventoryInterval/DI_synthetic_data_day_' + str(i) + '_ins2.csv')
        self.tar_all = df_inter.loc[:, self.index_tar]  # stations * periods
        self.tar_all = array(self.tar_all) / 100
        self.tmstamp = self.start_time  # starting decision epoch
        self.dv = [20] * self.V  # initial inventory of vehicles
        target_station = zeros((self.V * self.num_station))
        for i in range(self.V):
            zero_vector = zeros((self.num_station))
            index = 1  # np.random.randint(0, self.num_station-1)
            zero_vector[index] = 1
            target_station[i * self.num_station: (i + 1) * self.num_station] = zero_vector
        self.loc_v = target_station
        self.tar_loc = target_station
        self.est_tartime = [self.start_time, self.start_time + 1, self.start_time + 2,
                            self.start_time + 3]  # estimated arrival time for next target station
        observation = self._get_obs()
        self.w = []
        self.ongoing_reb = zeros((self.V))
        NS1 = arange(self.num_station)
        condition = ~isin(NS1, [index] * self.V)
        NS1 = NS1[condition]
        self.neighset = NS1
        info = self._get_info()
        return observation, info

    ##################################################################################
    # Convert obaservation space and action space to rebalacing operations
    ##################################################################################
    def obser_action_real(self, observation, action):
        station_index = [i for i in range(len(observation)) if list(observation)[i] == 1]
        current_station = station_index[0: self.V]
        target_station = station_index[self.V: 2 * self.V]
        for v in range(self.V):
            current_station[v] = (current_station[v] - self.num_station * v)
            target_station[v] = (target_station[v] - self.num_station * (v + self.V))
        dv = around(array(observation[self.V * self.num_station * 2: self.V * (
                    self.num_station * 2 + 1)]) * self.cap_V)  # current inventory of vehicles
        Vehicle_eta = array(observation[self.V * (self.num_station * 2 + 1): self.V * (
                    self.num_station * 2 + 2)]) * 250 + self.start_time
        ds = around(array(observation[-self.num_station - 1: -1]) * array(self.cap_s))
        start_min = around(observation[-1]* 250 + self.start_time)
        current_v = argmin(Vehicle_eta)  # find the min(eta), the current_v departs
        current_s = target_station[current_v]
        current_station[current_v] = current_s
        target_station[current_v] = action
        self.ongoing_reb[current_v] = -1
        # current_vehicle_inven = dv[current_v]
        # current_cap = self.cap_s[current_s]
        # current_inv = ds[current_s]
        # if station_level == 0:
        #     tar_bike = round(0.1 * current_cap)
        # elif station_level == 1:
        #     tar_bike = round(0.5 * current_cap)
        # else:
        #     tar_bike = round(0.9 * current_cap)
        # if tar_bike >= current_inv:  # drop off bikes-
        #     num_operation = -min(tar_bike - current_inv, current_vehicle_inven)
        # else:
        #     num_operation = min(current_inv - tar_bike, self.cap_V[current_v] - current_vehicle_inven)  # pick up
        Vehicle_eta[current_v] += int(max(self.Tran_tim[current_s][target_station[current_v]],
                                                      self.Tran_tim[target_station[current_v]][current_s]))

        return start_min, current_s, current_v, ds, dv, Vehicle_eta, target_station, current_station

    def simulation_singleV(self, info, observation, action):
        index_trip = []
        rrental = zeros((self.num_station))  # successful rentals for each station
        rreturn = zeros((self.num_station))
        start_min, current_s, current_v, ini_ds, ini_v, Vehicle_eta, target_station, current_station = self.obser_action_real(
            observation, action)  # drop-off -, pick-up +
        demand_request = info['Trips']
        w = info['Ongoing_oper']
        Ongoing_reb = self.ongoing_reb
        # print('current_station', current_station)
        # print('target_station',  target_station)
        est_time = min(Vehicle_eta)
        if Ongoing_reb[argmin(Vehicle_eta)] < 0:    # on the way
            a = [est_time-1, target_station[argmin(Vehicle_eta)], 5, argmin(Vehicle_eta)]
            if a not in w:
                w.append(a)
        # print('Station, Vehicle, Operation', [current_s + 1, current_v + 1, oper])
        # put demand into waiting list
        for k in range(len(demand_request)):
            if est_time > demand_request[k][0] >= start_min:
                index_trip.append(k)
                w.append(demand_request[k][0:2] + [1] + [0])  # waiting list rental 1 return 2 pick 3 drop 4 [T, S, 1, V]     index from 0
            # elif start_min <= demand_request[k][2] <= est_time:
            #     w.append(demand_request[k][2:] + [2] + [0])
        ds = copy(ini_ds)
        dv = copy(ini_v)
        count = 0
        lost_rental = 0
        lost_return = 0
        w.sort()
        if w != []:
            request = w[0]
            while request[0] < est_time and w != []:
                sign = 0
                time_min = request[0]
                station = int(request[1])  # index from 0
                vehicle = int(request[3])
                if request[2] == 1:  # rental
                    if ds[station] > 0:  # station inventory
                        ds[station] -= 1
                        w.append(demand_request[index_trip[count]][2:] + [2] + [0])
                        rrental[station] += 1
                    else:
                        lost_rental += 1
                    count += 1
                    w.remove(request)
                elif request[2] == 2:  # return
                    if self.cap_s[station] > ds[station]:
                        ds[station] += 1
                        rreturn[station] += 1
                    else:
                        lost_return += 1
                        lists = list(range(self.num_station))
                        lists.remove(station)
                        snear = self.NS[station][1]
                        while self.cap_s[snear] <= ds[snear]:
                            lists.remove(snear)
                            snear = next((x for x in self.NS[snear, :] if x in lists), None)
                        ds[snear] += 1
                    w.remove(request)
                elif request[2] == 3:  # pick up
                    if ds[station] > 0 and dv[vehicle] < self.cap_V[vehicle]:  # station inventory
                        ds[station] -= 1
                        dv[vehicle] += 1
                        Ongoing_reb[vehicle] -= 1
                        if Ongoing_reb[vehicle] == 0:  # rebalance finish
                            sign = 1
                        w.remove(request)
                    else:
                        w = list(filter(lambda x: x[2] != 3 or x[3] != vehicle,
                                        w))  # as long as the station is empty or vehicle is full, rebalancing stops
                        sign = 1
                elif request[2] == 4:  # drop off
                    if self.cap_s[station] > ds[station] and dv[vehicle] > 0:
                        ds[station] += 1
                        dv[vehicle] -= 1
                        Ongoing_reb[vehicle] -= 1
                        if Ongoing_reb[vehicle] == 0:  # rebalance finish
                            sign = 1
                        w.remove(request)
                    else:
                        w = list(filter(lambda x: x[2] != 4 or x[3] != vehicle, w))
                        sign = 1
                # The rebalancing finishes
                else:
                    #current_station[vehicle] = station
                    dv_re = dv / self.cap_V
                    est_tartime_re = (Vehicle_eta - self.start_time) / 250
                    normalds_re = ds / array(self.cap_s)
                    tmstamp1_re = (time_min +1 - self.start_time) / 250
                    hot_current = zeros((self.V * self.num_station))
                    hot_target = zeros((self.V * self.num_station))
                    for i in range(self.V):
                        hot_current[i * self.num_station + current_station[i]] = 1
                        hot_target[i * self.num_station + target_station[i]] = 1
                    loc_v_re = hot_current
                    loc_v_tar = hot_target
                    obs = hstack((loc_v_re, loc_v_tar, dv_re, est_tartime_re, normalds_re,tmstamp1_re))
                    obs.astype(float32)
                    q_values = self.model(torch.Tensor(obs).to(self.device))
                    action_re = torch.argmax(q_values).item()
                    #action_re = random.randint(0, 2)
                    if action_re == 0:
                        tar_bike = round(0.1 * self.cap_s[station])
                    elif action_re == 1:
                        tar_bike = round(0.5 * self.cap_s[station])
                    else:
                        tar_bike = round(0.9 * self.cap_s[station])
                    #time_window = int((time_min +1 - self.start_time) / 60)
                    #tar_bike = round(info['tar'][station, time_window] * self.cap_s[station])
                    if tar_bike >= ds[station]:  # drop off bikes-
                        num_operation = -min(tar_bike - ds[station], dv[vehicle])
                    else:
                        num_operation = min(ds[station] - tar_bike, self.cap_V[vehicle] - dv[vehicle])  # pick up
                    if num_operation > 0:
                        Ongoing_reb[vehicle] = num_operation
                        for j in range(int(num_operation)):
                            w.append([j * self.opert + time_min + 1] + [station] + [3] + [vehicle])  # pick by vehicle
                    elif num_operation < 0:
                        Ongoing_reb[vehicle] = -num_operation
                        for j in range(int(-num_operation)):
                            w.append([j * self.opert + time_min + 1] + [station] + [4] + [vehicle])  # drop by vehicle
                    else:
                        Ongoing_reb[vehicle] = 0
                    Vehicle_eta[vehicle] = time_min +1 + abs(num_operation) * self.opert
                    #print(Vehicle_eta, abs(num_operation))
                    if min(Vehicle_eta) > est_time:
                        for k in range(len(demand_request)):
                            if min(Vehicle_eta) > demand_request[k][0] >= est_time:
                                index_trip.append(k)
                                w.append(demand_request[k][0:2] + [1] + [0])
                    elif min(Vehicle_eta) < est_time:
                        w = [elem for elem in w if elem[0] < min(Vehicle_eta) or elem[2] != 1]
                    est_time = min(Vehicle_eta)
                    if Ongoing_reb[argmin(Vehicle_eta)] < 0:  # on the way
                        w.append([est_time - 1, target_station[argmin(Vehicle_eta)], 5, argmin(Vehicle_eta)])
                    w.remove(request)
                    #print(est_time, w)
                if sign == 1:
                    #print('rebal done', time_min, vehicle, Ongoing_reb[vehicle])
                    Vehicle_eta[vehicle] = time_min +1
                    Ongoing_reb[vehicle] = 0
                    if Vehicle_eta[vehicle] < est_time:
                        est_time = Vehicle_eta[vehicle]
                        w = [elem for elem in w if elem[0] < est_time or elem[2] != 1]
                        #print(index_trip)
                w.sort()
                if w != []:
                    request = w[0]
        reward = -lost_rental - lost_return
        return reward, dv, ds, est_time, w, Vehicle_eta, current_station, target_station, Ongoing_reb

    def step(self, action):
        # It accepts an action, computes the state of the environment after applying that action
        # get current status
        # simulate demand to get immediate reward
        observation = self._get_obs()
        info = self._get_info()
        # print('1', observation)
        # print('action', action)
        reward, self.dv, self.ds, end_min, self.w, self.est_tartime, current_station, target_station, self.ongoing_reb = self.simulation_singleV(
            info, observation, action)
        NS1 = arange(self.num_station)
        condition = ~isin(NS1, list(target_station)+list(current_station))
        NS1 = NS1[condition]
        self.neighset = NS1
        one_hot_current = zeros((self.V * self.num_station))
        one_hot_target = zeros((self.V * self.num_station))
        for i in range(self.V):
            one_hot_current[i * self.num_station + current_station[i]] = 1
            one_hot_target[i * self.num_station + target_station[i]] = 1
        self.loc_v = one_hot_current
        self.tar_loc = one_hot_target
        self.tmstamp = end_min
        # import sys
        # sys.exit()
        terminated = end_min >= self.end_time
        # update environment
        observation = self._get_obs()
        info = self._get_info()
        # print('2', observation)
        return observation, reward, terminated, False, info

    def render(self):
        pass

    def close(self):
        pass
