import gymnasium as gym
from gymnasium import spaces
from numpy import *
import json
import random
import torch
import torch.nn as nn
from numpy.random import choice
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
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(549, 1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            #nn.Linear(512,256),
            #nn.ReLU(),
            nn.Linear(512, 60),
        )
    def forward(self, x):
        return self.network(x)

class BssRebaE(gym.Env):
    ###########################################################################
    ###Define Environment
    ###########################################################################
    def __init__(self):
        self.accreward = 0
        self.stateindex = 0
        self.num_station = 60
        self.V = 4
        self.cap_s = [40] * 9 + [20] * 51
        self.opert = 2  # operation time in min
        with open('/local_workspace/lianjiaq/Instances/Ins_Pro2/dis_gen60.json') as fp:
            dis = json.load(fp)
        dis = array(dis)
        self.Tran_tim = 2 * dis
        with open('/local_workspace/lianjiaq/Instances/Ins_Pro2/NS_gen60.json') as fp:
            NS = json.load(fp)
        self.NS = array(NS)
        with open('/local_workspace/lianjiaq/Instances/Ins_Pro2/base2t.json') as fp:
            self.ini_ds = json.load(fp) # initial inventory
        #self.ini_ds = [40] * 9 + [6] *51
        self.cap_V = [40]*self.V
        self.start_time = 420
        self.end_time = 660     # in minutes
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
        self.action_space = spaces.Discrete(3)  # three levels to rebalance the urgent stations
        env_id = 'BssRouTM-v0'
        capture_video = True
        run_name = 'BssRouTM-v0__DQN_Mask__1__1711555083'
        #run_name = 'BssRouTM-v0__DQN_Mask__1__1706056411'
        self.device = torch.device("cpu")
        model_path = 'runs/BssRouTM-v0__DQN_Mask__1__1786422939/DQN_Mask.cleanrl_model'
        #model_path = 'runs/BssRouTM-v0__DQN_Mask__1__1715811877/DQN_Mask.cleanrl_model'
        #model_path = 'runs/BssRouTM-v0__DQN_Mask__1__1706396233/DQN_Mask.cleanrl_model'
        #envs = gym.vector.SyncVectorEnv([self.make_env(env_id, 0, capture_video, run_name)])
        self.model = QNetwork().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

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
        obs = hstack((self.loc_v, self.tar_loc, self.normaldv, self.normal_est_tartime, self.normalds, self.tmstamp1))
        return obs.astype(float32)

    def _get_info(self):
        return {'Trips': self.demand_request, 'Ongoing_oper': self.w, 'Ongoing_reb': self.ongoing_reb}


    def reset(self, seed=None, options= None):
        if seed is not None:
            random.seed(seed)
        self.ds = self.ini_ds
        # i = random.randint(0, 149)
        #i = 50
        i = random.choice([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                           67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
                           84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99])
        with open('/local_workspace/lianjiaq/Instances/Ins_Pro2/trip' + str(i+350) + '.json') as fp:
            self.demand_request = json.load(fp)
        print('Day:', i)
        self.tmstamp = self.start_time    # starting decision epoch
        self.dv = [20] * self.V       # initial inventory of vehicles
        target_station = zeros((self.V * self.num_station))
        for i in range(self.V):
            zero_vector = zeros((self.num_station))
            index = 1
            zero_vector[index] = 1
            target_station[i * self.num_station: (i + 1) * self.num_station] = zero_vector
        self.loc_v = target_station
        self.tar_loc = target_station
        self.est_tartime = [self.start_time, self.start_time + 1,self.start_time + 2,self.start_time + 3]  # estimated arrival time for next target station
        observation = self._get_obs()
        self.w = []
        self.ongoing_reb = -ones((self.V))
        info = self._get_info()
        self.accreward = 0
        self.stateindex = 0
        return observation, info

    def heuristic_route(self, ds, dv, NS, current_s, current_v, alpha):
        #distance
        uniform_distance_distributions = []
        sum_dis = 0
        for j in range(self.num_station):
            distance_ij = NS[min(j, current_s), max(current_s, j)]
            if distance_ij != 0:
                sum_dis += 1 / distance_ij
        for j in range(self.num_station):
            distance_ij = NS[min(j, current_s), max(current_s, j)]  # Get the distance between stations i and j
            if distance_ij == 0:
                distribution = 0.0  # Assign 0 distribution if distance is 0 to avoid division by zero
            else:
                distribution = (1 / distance_ij)/sum_dis # Calculate 1/distance
            uniform_distance_distributions.append(distribution)
        #capacity
        inven_level_v = dv[current_v]/self.cap_V[current_v]
        inven_level_s = ds/self.cap_s
        inven_distribution = (ones((self.num_station))- inven_level_s) * inven_level_v + inven_level_s * (1-inven_level_v)
        uniform_inven_distribution = inven_distribution/ sum(inven_distribution)
        total_distribution = alpha * array(uniform_distance_distributions) + (1-alpha) * uniform_inven_distribution

        return total_distribution

    def heuristic_route1(self, ds, dv, NS, current_s, current_v, alpha, m):
        # Distance
        uniform_distance_distributions = []
        sum_dis = 0
        for j in range(self.num_station):
            distance_ij = NS[min(j, current_s), max(current_s, j)]
            if distance_ij != 0:
                sum_dis += (1 / distance_ij) ** m
        for j in range(self.num_station):
            distance_ij = NS[min(j, current_s), max(current_s, j)]
            if distance_ij == 0:
                distribution = 0.0
            else:
                distribution = (1 / distance_ij) ** m / sum_dis  # Calculate 1/distance with power m
            uniform_distance_distributions.append(distribution)
        # Capacity
        inven_level_v = dv[current_v] / self.cap_V[current_v]
        inven_level_s = ds / self.cap_s
        inven_distribution = ((ones((self.num_station)) - inven_level_s) * inven_level_v + inven_level_s * (1 - inven_level_v)) ** m
        sum_inven_distribution = sum(inven_distribution)
        uniform_inven_distribution = inven_distribution / sum_inven_distribution
        total_distribution = alpha * array(uniform_distance_distributions) + (1 - alpha) * uniform_inven_distribution

        return total_distribution
 
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
        dv = around(array(observation[self.V * self.num_station * 2: self.V * (self.num_station * 2 + 1)]) * self.cap_V)  # current inventory of vehicles
        #print(dv)
        Vehicle_eta = array(observation[self.V * (self.num_station * 2 + 1) : self.V * (self.num_station * 2 + 2)]) * 250 + self.start_time
        ds = around(array(observation[-self.num_station - 1: -1]) * array(self.cap_s))
        #print(Vehicle_eta)
        #print(ds)
        start_min = min(Vehicle_eta)
        current_v = argmin(Vehicle_eta)  # find the min(eta), the current_v arrives
        current_s = target_station[current_v]
        current_station[current_v] = current_s
        station_level = int(action)
        current_vehicle_inven = dv[current_v]
        current_cap = self.cap_s[current_s]
        current_inv = ds[current_s]
        if station_level == 0:
            tar_bike = round(0.1 * current_cap)
        elif station_level == 1:
            tar_bike = round(0.5 * current_cap)
        else:
            tar_bike = round(0.9 * current_cap)
        if tar_bike >= current_inv:  # drop off bikes-
            num_operation = -min(tar_bike - current_inv, current_vehicle_inven)
        else:
            num_operation = min(current_inv - tar_bike, self.cap_V[current_v] - current_vehicle_inven)  # pick up
        Vehicle_eta[current_v] += self.opert * abs(num_operation)
        self.ongoing_reb[current_v] = abs(num_operation)
        return start_min, num_operation, current_s, current_v, ds, dv, Vehicle_eta, target_station, current_station

    def simulation_singleV(self, info, observation, action):
        demand_request = info['Trips']
        w = info['Ongoing_oper']
        Ongoing_reb = self.ongoing_reb
        index_trip = []
        rrental = zeros((self.num_station))  # successful rentals for each station
        rreturn = zeros((self.num_station))
        start_min, oper, current_s, current_v, ini_ds, ini_v, Vehicle_eta, target_station, current_station= self.obser_action_real(observation, action)   # drop-off -, pick-up +
        #print('state index', self.stateindex)
        #print('action', action)
        #print('current_station', current_station)
        #print('target_station',  target_station)
        #print('eta', Vehicle_eta)
        #print('rebalancing', Ongoing_reb)
        #print('observation', observation)
        #print('station inven', ini_ds)
        est_time = min(Vehicle_eta)
        #print('Station, Vehicle, Operation', [current_s , current_v , oper])
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
        lost_rental=0
        lost_return = 0
        if oper > 0:
            for j in range(int(oper)):
                w.append([j * self.opert + start_min] + [current_s] + [3] + [current_v])  # pick by vehicle
        elif oper < 0:
            for j in range(int(-oper)):
                w.append([j * self.opert + start_min] + [current_s] + [4] + [current_v])  # drop by vehicle
        else:
            w.append([est_time - 1, current_s, 5, current_v])
        w.sort()
        #print('demand', w)
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
                elif request[2] == 4 :  # drop off
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
                    w.remove(request)
                    sign = 1
                if sign == 1:
                    #print('time_min',time_min)
                    dv_re = dv / self.cap_V
                    Vehicle_eta[vehicle] = time_min + 1
                    est_tartime_re = (Vehicle_eta - self.start_time) / 250
                    normalds_re = ds / array(self.cap_s)
                    tmstamp1_re = (time_min + 1 - self.start_time) / 250
                    hot_current = zeros((self.V * self.num_station))
                    hot_target = zeros((self.V * self.num_station))
                    for i in range(self.V):
                        hot_current[i * self.num_station + current_station[i]] = 1
                        hot_target[i * self.num_station + target_station[i]] = 1
                    loc_v_re = hot_current
                    loc_v_tar = hot_target
                    obs = hstack((loc_v_re, loc_v_tar, dv_re, est_tartime_re, normalds_re, tmstamp1_re))
                    obs.astype(float32)
                    self.stateindex += 1
                    #print('state index', self.stateindex)
                    q_values = self.model(torch.Tensor(obs).to(self.device))
                    q_values[list(current_station) + list(target_station)] = -100000
                    #print('Mask station', current_station, target_station)
                    #print('Q_mask', q_values)
                    action_route = torch.argmax(q_values).item()
                    #ranking = self.heuristic_route1(ds, dv, self.Tran_tim, station, vehicle, 0.8,1)
                    #ranking[list(target_station)+list(current_station)] = 0
                    #probabilities = ranking /sum(ranking)
                    #action_route = choice(len(ranking), p= probabilities)
                    #action_route = argmax(ranking)
                    #print('vehicle routing', vehicle, action_route)
                    #print('observation',obs)
                    #print('station inven', ds)
                    #action_route = random.randint(0, self.num_station-1)
                    #list_urgent = []
                    #for i in range(self.num_station):
                    #    list_urgent.append(ds[i] / self.cap_s[i])
                    #list_urgent = array(list_urgent)
                    #if dv[vehicle] / self.cap_V[vehicle] > 0.5:
                    #    list_urgent[station] = 1
                    #    action_route = argmin(list_urgent)
                    #else:
                    #    list_urgent[station] = 0
                    #    action_route = argmax(list_urgent)
                    #print('distance', max(self.Tran_tim[current_station[vehicle]][action_route],
                    #        self.Tran_tim[action_route][current_station[vehicle]]))
                    if max(self.Tran_tim[current_station[vehicle]][action_route],
                            self.Tran_tim[action_route][current_station[vehicle]]) <= 1:
                        #print('short route between',current_station[vehicle], action_route)
                        Vehicle_eta[vehicle] += 1
                    else:
                        Vehicle_eta[vehicle] += int(max(self.Tran_tim[current_station[vehicle]][action_route],
                            self.Tran_tim[action_route][current_station[vehicle]]))

                    #Vehicle_eta[vehicle] = time_min + int(max(self.Tran_tim[current_station[vehicle]][action_route],
                    #        self.Tran_tim[action_route][current_station[vehicle]]))
                    target_station[vehicle] = action_route
                    Ongoing_reb[vehicle] = -1
                    #print(Vehicle_eta, action_route)
                    if min(Vehicle_eta) > est_time:
                        for k in range(len(demand_request)):
                            if min(Vehicle_eta) > demand_request[k][0] > est_time:
                                index_trip.append(k)
                                w.append(demand_request[k][0:2] + [1] + [0])
                        est_time = min(Vehicle_eta)
                    else:
                        est_time = min(Vehicle_eta)
                        w = [elem for elem in w if elem[0] < est_time or elem[2] != 1]
                    #print('eta', est_time, 'remain demand',w)
                w.sort()
                if w != []:
                    request = w[0]
        #print('demand left', w)
        reward = -lost_rental - lost_return
        #print('est', est_time)
        return reward, dv, ds, est_time, w, Vehicle_eta, current_station, target_station, Ongoing_reb
    def step(self, action):
        # It accepts an action, computes the state of the environment after applying that action
        # get current status
        # simulate demand to get immediate reward
        observation = self._get_obs()
        info = self._get_info()
        #print('1', observation)
        #print('action', action)
        reward, self.dv, self.ds, end_min, self.w, self.est_tartime, current_station, target_station, self.ongoing_reb = self.simulation_singleV(info, observation, action)
        #print('current_station after step', current_station)
        #print('target_station after step', target_station)
        #print('rebalancing', self.ongoing_reb)
        one_hot_current = zeros((self.V * self.num_station))
        one_hot_target = zeros((self.V * self.num_station))
        for i in range(self.V):
            one_hot_current[i*self.num_station + current_station[i]] = 1
            one_hot_target[i*self.num_station + target_station[i]] = 1
        self.loc_v = one_hot_current
        self.tar_loc = one_hot_target
        self.tmstamp = end_min
        # import sys
        # sys.exit()
        terminated = end_min >= self.end_time
        # update environment
        self.stateindex += 1
        #print('reward',reward)
        self.accreward += reward
        #print('accumulated reward', self.accreward)
        observation = self._get_obs()
        # info = self._get_info()
        #print('2', observation)
        return observation, reward, terminated, False, info

    def render(self):
        pass
    def close(self):
        pass
