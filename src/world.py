import pandas as pd
import datetime
import util
from map import *
from agent import *

class WorldState:
    def __init__(self, _map = Mymap()):
        self.map = _map
        self.agent = ApproximateQAgent(self.getLegalActions)
        self.agent_pos = (5,2) # at campus
        self.curr_time = datetime.datetime(2020,1,1,11)
        self.end_time = datetime.datetime(2020,1,1,14)  # 11:00 --- 14:00, 1 min per action
        self.delta_t = datetime.timedelta(seconds=60)    # deltat = 1 min
        self.orders_packet_idx = 0

    def getImmOrders(self, _time):
        curr_time_string = _time.strftime("%H:%M:%S")
        df = pd.read_csv("./data/" + str(self.orders_packet_idx) + ".csv")
        ImmOrders = df[df['created_time'] == curr_time_string].values.tolist()
        return ImmOrders

    def getLegalActions(self, _state):
        """
        state = (pos, time, [received], [carrying])
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
        if (_state[0][0]-1, _state[0]s[1]) in successors:
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



        
        

        