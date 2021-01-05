import random as rd
import datetime as dt
import util
# from buffer import *
from featureExtractor import *

class ApproximateQAgent:
    """
    Approximate Q-learning agent
    Use a particular feature extractor
    """
    def __init__(self, _actionFn, _world):
        self.alpha = 0.1  # learning rate
        self.gamma = 0.8  # discounting factor
        self.epsilon = 0.05  # exploration factor
        self.actionFn = _actionFn

        self.orderBuffer = []
        self.featExtractor = FeatureExtractor(_world)
        self.weights = util.Counter()
        self.world = _world
        self.policy = util.Counter()  # Contain the action idx

    def getWeights(self):
        return self.weights
    
    def getLegalActions(self, _state):
        return self.actionFn(_state)

    def getQValue(self, _state, _action):
        """
        Should return Q(state, action) = w * featureVector
        where * is the dotProduct operator
        """
        return self.getWeights() * self.featExtractor.getFeatures(_state, _action)
    
    def getValue(self, _state):
        """
        Returns max_action Q(state, action)
        where the max is over legal actions.
        Note that if there are no legal actions, 
        which is the case at the terminal state, should return 0.0
        """
        if len(self.getLegalActions(_state)) == 0:
            return 0.0
        qvalues = [self.getQValue(_state, action) for action in self.getLegalActions(_state)]
        # if dt.datetime.now().second % 10 == 0:
            # print(qvalues)
        return max(qvalues)

    def getPolicy_byDict(self, _state):
        actions = self.getLegalActions(_state)
        return actions[self.policy[_state]]

    def getPolicy_byQvalues(self, _state):
        """
        Use approximate Q learning
        return argmax_action Q(state, action)
        """
        if len(self.getLegalActions(_state)) == 0:
            return None
        max_qvalue = self.getValue(_state)
        best_actions = [action for action in self.getLegalActions(_state) if self.getQValue(_state, action) == max_qvalue]
        if len(best_actions) == 0:
            print("max_qvalue: ", max_qvalue)
        return rd.choice(best_actions)

    def getPolicy_byValues(self, _state, _values):
        """
        values = util.Counter
        """
        actions = self.getLegalActions(_state)
        assert(len(actions) != 0)
        successors = []
        for action in actions:
            successor, _ = self.world.getSuccessorStateandReward(_state, action)
            successors.append(successor)
        highest_value = max([_values[successor] for successor in successors])
        highest_value_states = [successor for successor in successors if _values[successor] == highest_value]
        corres_actions = [action.index(highest_state) for highest_state in highest_value_states]
        return rd.choice(corres_actions)

    def getAction_byDict(self, _state):
        """
        a random action or policy
        """
        actions = self.getLegalActions(_state)
        if len(actions) == 0:
            return None
        if util.flipCoin(self.epsilon):
            return rd.choice(actions)
        else:
            return self.getPolicy_byDict(_state)
    
    def getAction_byQvalues(self, _state):
        """
        a random action or policy_Qvalues
        """
        actions = self.getLegalActions(_state)
        if len(actions) == 0:
            return None
        if util.flipCoin(self.epsilon):
            return rd.choice(actions)
        else:
            return self.getPolicy_byQvalues(_state)

    def update(self, _state, _action, _nextState, _reward):
        """
        Should update weights based on transition
        """
        diff = _reward + self.gamma * self.getValue(_nextState) - self.getQValue(_state, _action)
        for key in self.featExtractor.getFeatures(_state, _action):
            self.weights[key] += self.alpha * diff * self.featExtractor.getFeatures(_state, _action)[key]

