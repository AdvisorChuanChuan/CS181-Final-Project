import pandas as pd
import datetime as dt
import util
from map import *
from agent import *

class WorldState:
    def __init__(self, _map = Mymap()):
        self.map = _map
        self.agent = ApproximateQAgent(self.getLegalActions)
        self.agent_pos = (5,2) # at campus
        self.curr_time = dt.datetime(2020,1,1,11)
        self.end_time = dt.datetime(2020,1,1,14)  # 11:00 --- 14:00, 1 min per action
        self.delta_t = dt.timedelta(seconds=60)    # deltat = 1 min
        self.orders_packet_idx = 0
        self.training_df = pd.read_csv("./data/" + str(self.orders_packet_idx) + ".csv")

        self.living_cost = -1
        self.reward_per_order = 10


    def getImmOrders(self, _str_time):
        """
        Return the list of immediate orders
        """
        curr_time_string = _str_time
        ImmOrders = self.training_df[self.training_df['created_time'] == curr_time_string].values.tolist()
        return ImmOrders

    def getSuccessorStateandReward(self, _state, _action):
        """
        * state = (pos, str_time, [received], [carrying])
        * action = ('South', [0,1], [2])

        """
        nextState = []
        # Find next position
        if _action[0] == 'East':
            nextState.append((_state[0][0], _state[0][1]+1))
        elif _action[0] == 'West':
            nextState.append((_state[0][0], _state[0][1]-1))
        elif _action[0] == 'South':
            nextState.append((_state[0][0]+1, _state[0][1]))
        elif _action[0] == 'North':
            nextState.append((_state[0][0]-1, _state[0][1]))
        else:
            nextState.append(_state[0])
        # Find next time
        dt_curr_time = util.str_to_datetime(_state[1])
        dt_next_time = dt_curr_time + self.delta_t
        nextState.append(util.datetime_to_str(dt_next_time))
        # Receive orders
        nextState.append(_state[2].copy())
        for received_idx in _action[1]:
            nextState[2].append(self.training_df.iloc[received_idx].values.tolist())
        # Update received orders
        nextState.append(_state[3].copy())
        if _state[0] in self.map.restaurants_poss:
            cur_res_idx = self.map.restaurants_poss.index(_state[0])
            cur_res_name = 'res_' + chr(ord('A') + cur_res_idx)
            for received_order in nextState[2]:
                if cur_res_name in received_order:
                    nextState[3].append(received_order.copy())
                    nextState[2].remove(received_order)
        # Update carrying orders
        reward = self.living_cost
        if _state[0] == self.map.des_pos:
            reward = self.reward_per_order * len(nextState[3])
            nextState[3] = []
        # TODO: add penalty for delayed orders
        return nextState, reward



    def getLegalActions(self, _state):
        """
        state = (pos, str_time, [received], [carrying])
        Get the legal actions for the agent:
        * Move east/west/south/north
        * Receive/Reject orders
        * Stay down and do nothing
        e.g. an action ('South', [0,1], [2]) means to go south, while receiving order 0&1 but rejecting 2
        """
        if _state[1] >= self.end_time:
            return []
        
        successors = self.map.get_successor(_state[0])
        assert(len(successors)>0)
        move_actions = ['Stay']
        if (_state[0][0]+1, _state[0][1]) in successors:
            move_actions.append('South')
        if (_state[0][0]-1, _state[0][1]) in successors:
            move_actions.append('North')
        if (_state[0][0], _state[0][1]+1) in successors:
            move_actions.append('East')
        if (_state[0][0], _state[0][1]-1) in successors:
            move_actions.append('West')
        
        ImmOrders = self.getImmOrders(_state[1])
        order_actions = util.getHandleOrdersChoices(ImmOrders)
        actions = []
        for move_action in move_actions:
            for order_action in order_actions:
                actions.append((move_action, order_action[0], order_action[1]))
        
        return actions



        
        

        