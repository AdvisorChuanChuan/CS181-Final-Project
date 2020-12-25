import util

class WorldState:
    def __init__(self):
        self.destination_idx = 0
        self.restaurants_num = 4
        self.restaurants_idxs = [i for i in range(1,self.restaurants_num+1)]
        self.agent_pos = 0
        self.curr_time = 0
        self.end_time = 180  # 11:00 --- 14:00

    def getLegalActions(self):
        """
        Get the legal actions for the agent:
        * Go to location i (anywhere other than current location)
        * Spend some time choosing orders
        """
        if self.curr_time >= self.end_time:
            return []
        

        