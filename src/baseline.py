from map import *
import datetime as dt
import pandas as pd

class GreedyAgent:
    def __init__(self):
        self.received = []
        self.carrying = []
        self.map = Mymap()
        self.ddl = dt.datetime(2020, 1, 2, 13, 00)
        self.pos = (5, 2)

    def getAction(self, pos, restaurants):
        next_restaurant, whole_path_len = self.map.search_route(restaurants, pos)
        if next_restaurant is None:
            next_res_pos = self.map.des_pos
        else:
            next_res_pos = restaurants[next_restaurant]
        next_pos = self.map.bfs(pos, next_res_pos)
        return next_pos

    def update_ddl(self):
        ddl_list = []
        for carry in self.carrying:
            late_time = dt.timedelta(minutes=carry[3])
            ddl_list.append(util.str_to_datetime(carry[1]) + late_time)
        if len(self.carrying) == 0:
            self.ddl = dt.datetime(2020, 1, 2, 13, 00)
        else:
            self.ddl = min(ddl_list)


class GreedyWorld:
    def __init__(self):
        self.map = Mymap()
        self.agent = GreedyAgent()
        self.start_time = dt.datetime(2020, 1, 1, 11, 30)
        self.end_time = dt.datetime(2020, 1, 1, 13, 00)
        self.delta_t = dt.timedelta(seconds=60)
        self.orders_packet_idx = 0
        self.training_df = pd.read_csv("data/" + str(self.orders_packet_idx) + ".csv")[0:15]
        self.res_name = ['res_A', 'res_B', 'res_C', 'res_D']
        self.reward = 0
        self.living_cost = -1
        self.reward_per_order = 10
        self.penalty_per_order = 9
        self.max_received_num = 3
        self.max_carrying_num = 3

    def getImmOrders(self, _str_time):
        """
        Return the list of immediate orders
        """
        curr_time_string = _str_time
        curr_df = self.training_df[self.training_df['created_time'] == curr_time_string]
        ImmOrders = tuple([tuple(x) for x in curr_df.values])
        return ImmOrders

    def check_in_time(self, current_time):
        path_to_des = len(self.map.bfs(self.agent.pos, self.map.des_pos))
        time_to_des = dt.timedelta(minutes=(path_to_des + 2))
        left_time = self.agent.ddl - current_time
        return time_to_des < left_time

    def test_greedy(self):
        """
        Test the greedy agent's score
        """
        current_time = self.start_time
        while current_time < self.end_time:
            self.reward += self.living_cost
            current_time_str = util.datetime_to_str(current_time)
            ImmOrders = self.getImmOrders(current_time_str)
            # print(ImmOrders)
            while len(self.agent.received) + len(self.agent.carrying) < 5 and len(ImmOrders) > 0:
                self.agent.received.append(ImmOrders[0])
                ImmOrders = ImmOrders[1:]
            # print(self.agent.received)
            self.agent.received = self.agent.received
            if len(self.agent.received) + len(self.agent.carrying) != 0:
                " Judge whether the pos is on the destination "
                if self.agent.pos == self.map.des_pos:
                    self.reward += self.reward_per_order * len(self.agent.carrying)
                    for order in self.agent.carrying:
                        if current_time > util.getDueTime(order):
                            self.reward -= self.penalty_per_order
                    self.agent.carrying = []
                print(self.agent.received)
                print(len(self.agent.received))
                " Judge whether the pos is on the restaurant pos "
                for i in range(0, 4):
                    if self.agent.pos == self.map.restaurants_poss[i]:
                        # for j in range(len(self.agent.received)):
                        #     print(j)
                        #     order = self.agent.received[j]
                        for order in self.agent.received:
                            print(order)
                            if order[2] == self.res_name[i]:
                                print('here!')
                                self.agent.carrying.append(order)
                                self.agent.received.remove(order)
                print(current_time)
                print(self.agent.received)
                " Plan the next pos "
                self.agent.update_ddl()
                if not self.check_in_time(current_time):
                    self.agent.pos = (self.map.bfs(self.agent.pos, self.map.des_pos))[0]
                    # print(self.agent.pos)
                else:
                    if len(self.agent.received) != 0:
                        restaurants_pos = []
                        restaurants_name = []
                        for order in self.agent.received:
                            if order[2] not in restaurants_name:
                                restaurants_name.append(order[2])
                        for restaurant_name in restaurants_name:
                            restaurants_pos.append(self.map.restaurants_poss[self.res_name.index(restaurant_name)])
                        # print(restaurants_pos)
                        path_next_res, path_min = self.map.search_route(restaurants_pos, self.agent.pos)
                        # print(restaurants_pos)
                        # print(path_next_res)
                        # print(restaurants_pos[path_next_res])
                        next_pos = (self.map.bfs(self.agent.pos, restaurants_pos[path_next_res]))[0]
                        self.agent.pos = next_pos
                    elif len(self.agent.carrying) != 0:
                        self.agent.pos = (self.map.bfs(self.agent.pos, self.map.des_pos))[0]
                        # print(self.agent.pos)

            current_time += self.delta_t


basegreedy = GreedyWorld()
basegreedy.test_greedy()