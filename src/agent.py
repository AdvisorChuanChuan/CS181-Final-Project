import util
from buffer import *
from featureExtractor import *

class ApproximateQAgent:
    """
    Approximate Q-learning agent
    Use a particular feature extractor
    """
    def __init__(self, _extractor = FeatureExtractor()):
        self.alpha = 0.2  # learning rate
        self.gamma = 0.8  # discounting factor
        self.epsilon = 0.05  # exploration factor

        self.orderBuffer = OrderBuffer()

        self.featExtractor = _extractor
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights
    
    def getLegalActions(self, _state):
        return _state.getLegalActions()

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

        return max([self.getQValue(_state, action) for action in self.getLegalActions(_state)])

    def update(self, _state, _action, _nextState, _reward):
        """
        Should update weights based on transition
        """
        diff = _reward + self.gamma * self.getValue(_nextState) - self.getQValue(_state, _action)
        for key in self.featExtractor.getFeatures(_state, _action):
            self.weights[key] += self.alpha * diff * self.featExtractor.getFeatures(_state, _action)[key]

