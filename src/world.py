import pandas as pd
import datetime
import util
from map import *
from agent import *

class WorldState:
    def __init__(self, _map = Mymap()):
        self.map = _map
        self.agent = ApproximateQAgent()
        self.agent_pos = (5,2) # at campus
        self.curr_time = datetime.datetime(2020,1,1,11)
        self.end_time = datetime.datetime(2020,1,1,14)  # 11:00 --- 14:00, 1 min per action
        self.delta_t = datetime.timedelta(seconds=60)    # deltat = 1 min

        self.orders_packet_idx = 0
        self.immediate_orders = []  # orders at this curr_time

    def getImmOrders(self):
        curr_time_string = self.curr_time.strftime("%H:%M:%S")
        df = pd.read_csv("./data/" + str(self.orders_packet_idx) + ".csv")
        ImmOrders = df[df['created_time'] == curr_time_string].values.tolist()
        return ImmOrders

    def getLegalActions(self):
        """
        Get the legal actions for the agent:
        * Move east/west/south/north
        * Receive/Reject orders
        * Stay down and do nothing
        e.g. an action ('South', [0,1], [2]) means to go south, while receiving order 0&1 but rejecting 2
        """
        if self.curr_time >= self.end_time:
            return []
        
        successors = self.map.get_successor(self.agent_pos)
        assert(len(successors)>0)
        move_actions = ['Stay']
        if (self.agent_pos[0]+1, self.agent_pos[1]) in successors:
            move_actions.append('South')
        if (self.agent_pos[0]-1, self.agent_pos[1]) in successors:
            move_actions.append('North')
        if (self.agent_pos[0], self.agent_pos[1]+1) in successors:
            move_actions.append('East')
        if (self.agent_pos[0], self.agent_pos[1]-1) in successors:
            move_actions.append('West')
        
        ImmOrders = self.getImmOrders()
        order_actions = util.getHandleOrdersChoices(ImmOrders)
        actions = []
        for move_action in move_actions:
            for order_action in order_actions:
                actions.append((move_action, order_action[0], order_action[1]))
        
        return actions



        
        

        