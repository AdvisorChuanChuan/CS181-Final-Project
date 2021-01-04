# from world import World
import util

class FeatureExtractor:
    def __init__(self, _world):
        self.world = _world

    def getFeatures(self, _state, _action):
        """
        Extract some features from state-action pair
        * state = (pos, str_time, [received], [carrying])
        * action = ('South', [0,1], [2])

        """
        nextState,_ = self.world.getSuccessorStateandReward(_state, _action)
        features = util.Counter()

        features["#-receiving-orders-now"] = float(len(_action[1]))
        features["#-rejecting-orders-now"] = float(len(_action[2]))

        features["#-received-orders-1-step-away"] = float(len(nextState[2]))
        features["#-carring-orders-1-step-away"] = float(len(nextState[3]))

        features["dist-to-des-1-step-away"] = float(len(self.world.map.bfs(nextState[0], self.world.map.des_pos)))
        
        extra_mins_sum = 0
        late_orders_num = 0
        for order in nextState[2]+nextState[3]:
            if util.str_to_datetime(nextState[1]) < util.getDueTime(order):
                extra_time = util.getDueTime(order) - util.str_to_datetime(nextState[1])
                extra_mins_sum += extra_time.total_seconds() / 60
            else:
                late_orders_num += 1
        # features["extra-mins-sum-1-step-away"] = float(extra_mins_sum)
        features["late_orders_num-1-step-away"] = float(late_orders_num)

        return features