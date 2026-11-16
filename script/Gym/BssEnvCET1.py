import gymnasium as gym
from gymnasium import spaces
from numpy import *
import json
import random

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

class BssCTET1(gym.Env):
    ###########################################################################
    ###Define Environment
    ###########################################################################
    def __init__(self):
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
        self.cap_V = [40]*self.V
        self.start_time = 420
        self.end_time = 660        # in minutes
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
        self_station_level = 3
        self.neighbour = self.num_station - self.V
        self.action_space = spaces.Discrete(self_station_level * self.neighbour)  # three levels to rebalance the stations with stations

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
        return {'Trips': self.demand_request, 'Ongoing_oper': self.w, 'Ongoing_reb': self.ongoing_reb}

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
        self.ds = self.ini_ds
        #i = random.randint(0, 149)
        i = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                           19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                           36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 100, 101,
                           102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
                           116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
                           130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
                           144, 145, 146, 147, 148, 149])
        with open('/local_workspace/lianjiaq/Instances/Ins_Pro2/trip' + str(i+350) + '.json') as fp:
            self.demand_request = json.load(fp)
        print('Day', i+1)
        self.tmstamp = self.start_time    # starting decision epoch
        self.dv = [20] * self.V       # initial inventory of vehicles
        target_station = zeros((self.V * self.num_station))
        for i in range(self.V):
            zero_vector = zeros((self.num_station))
            index = 1  # np.random.randint(0, self.num_station-1)
            zero_vector[index] = 1
            target_station[i * self.num_station: (i + 1) * self.num_station] = zero_vector
        self.loc_v = target_station
        self.tar_loc = target_station
        self.est_tartime = [self.start_time, self.start_time + 1,self.start_time + 2,self.start_time + 3]  # estimated arrival time for next target station
        observation = self._get_obs()
        self.w = []
        self.ongoing_reb = zeros((self.V))
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
        dv = around(array(observation[self.V * self.num_station * 2: self.V * (self.num_station * 2 + 1)]) * self.cap_V)  # current inventory of vehicles
        Vehicle_eta = array(observation[self.V * (self.num_station * 2 + 1): self.V * (self.num_station * 2 + 2)]) * 250 + self.start_time
        ds = around(array(observation[-self.num_station - 1: -1]) * array(self.cap_s))
        start_min = min(Vehicle_eta)
        current_v = argmin(Vehicle_eta)  # find the min(eta), the current_v arrives
        current_s = target_station[current_v]
        current_station[current_v] = current_s
        neighbour = self.NS[current_s]
        condition = ~isin(neighbour, target_station)
        neighbour = neighbour[condition]
        neighbour = neighbour[:self.neighbour]
        target_station[current_v] = neighbour[action % self.neighbour]
        #target_station[current_v] = action % self.num_station
        station_level = int(action /self.neighbour)
        current_vehicle_inven = dv[current_v]
        current_cap = self.cap_s[current_s]
        current_inv = ds[current_s]
        if station_level == 0:
            tar_bike = round(0.0 * current_cap)
        elif station_level == 1:
            tar_bike = round(0.5 * current_cap)
        else:
            tar_bike = round(1 * current_cap)
        if tar_bike >= current_inv:  # drop off bikes-
            num_operation = -min(tar_bike - current_inv, current_vehicle_inven)
        else:
            num_operation = min(current_inv - tar_bike, self.cap_V[current_v] - current_vehicle_inven)  # pick up

        a = self.opert * abs(num_operation) + int(max(self.Tran_tim[current_s][target_station[current_v]],
                                                                          self.Tran_tim[target_station[current_v]][current_s]))
        if a == 0:
            Vehicle_eta[current_v] += 1
        else:
            Vehicle_eta[current_v] += a
        return start_min, num_operation, current_s, current_v, ds, dv, Vehicle_eta, target_station, current_station
    def simulation_singleV(self, info, observation, action):
        demand_request = info['Trips']
        w = info['Ongoing_oper']
        Ongoing_reb = info['Ongoing_reb']
        index_trip = []
        rrental = zeros((self.num_station))  # successful rentals for each station
        rreturn = zeros((self.num_station))
        start_min, oper, current_s, current_v, ini_ds, ini_v, Vehicle_eta, target_station, current_station= self.obser_action_real(observation, action)   # drop-off -, pick-up +
        #print('current_station', current_station)
        #print('target_station',  target_station)
        est_time = min(Vehicle_eta)
        # print('Station, Vehicle, Operation', [current_s + 1, current_v + 1, oper])
        Ongoing_reb[current_v] = abs(oper)
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
        if oper >= 0:
            for j in range(int(oper)):
                w.append([j * self.opert + start_min] + [current_s] + [3] + [current_v])  # pick by vehicle
        else:
            for j in range(int(-oper)):
                w.append([j * self.opert + start_min] + [current_s] + [4] + [current_v])  # drop by vehicle
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
                else :  # drop off
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
                if sign == 1:
                    Vehicle_eta[vehicle] = time_min + int(max(self.Tran_tim[current_station[vehicle]][target_station[vehicle]],
                                                          self.Tran_tim[target_station[vehicle]][current_station[vehicle]]))
                    Ongoing_reb[vehicle] = 0
                    if Vehicle_eta[vehicle] < est_time:
                        est_time = Vehicle_eta[vehicle]
                        w = list(filter(lambda x: x[2] != 1, w))
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
        #print('1', observation)
        #print('action', action)
        reward, self.dv, self.ds, end_min, self.w, self.est_tartime, current_station, target_station, self.ongoing_reb = self.simulation_singleV(info, observation, action)
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
        observation = self._get_obs()
        info = self._get_info()
        #print('2', observation)
        return observation, reward, terminated, False, info

    def render(self):
        pass
    def close(self):
        pass
