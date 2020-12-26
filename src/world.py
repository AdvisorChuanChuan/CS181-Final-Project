import util
from map import *
from agent import *

class WorldState:
    def __init__(self, _map = Mymap()):
        self.map = _map
        self.agent = ApproximateQAgent()
        self.agent_pos = (0,0)  # TODO
        self.curr_time = 0
        self.end_time = 180 * 2  # 11:00 --- 14:00, 0.5 min per action

    def getLegalActions(self):
        """
        Get the legal actions for the agent:
        * Move east/west/south/north
        * Receive/Reject orders
        * Stay down and do nothing
        """
        if self.curr_time >= self.end_time:
            return []
        

        