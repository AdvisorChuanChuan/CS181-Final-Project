import util

class ApproximateQAgent:
    """
    Approximate Q-learning agent
    Use a particular feature extractor
    """
    def __init__(self, extractor):
        self.alpha = 0.2  # learning rate
        self.gamma = 0.8  # discounting factor
        self.epsilon = 0.05  # exploration factor

        self.featExtractor = extractor
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights
    
    def getLegalActions(self, state):
        """
        TODO
        """

    def getQValue(self, state, action):
        """
        Should return Q(state, action) = w * featureVector
        where * is the dotProduct operator
        """
        return self.getWeights() * self.featExtractor.getFeatures(state, action)
    
    def getValue(self, state):
        """
        Returns max_action Q(state, action)
        where the max is over legal actions.
        Note that if there are no legal actions, 
        which is the case at the terminal state, should return 0.0
        """
        if len(self.getLegalActions(state)) == 0:
            return 0.0

        return max([self.getQValue(state, action) for action in self.getLegalActions(state)])

    def update(self, state, action, nextState, reward):
        """
        Should update weights based on transition
        """
        diff = reward + self.gamma * self.getValue(nextState) - self.getQValue(state, action)
        for key in self.featExtractor.getFeatures(state, action):
            self.weights[key] += self.alpha * diff * self.featExtractor.getFeatures(state, action)[key]

