import pandas as pd
import datetime as dt
import math
import random as rd
import util
from map import *
from agent import *

class World:
    def __init__(self, _map = Mymap()):
        self.map = _map
        self.agent = ApproximateQAgent(self.getLegalActions, self)
        self.init_agent_pos = (5,2) # at campus
        self.start_time = dt.datetime(2020,1,1,11,30)
        self.end_time = dt.datetime(2020,1,1,13,00)  # 11:00 --- 14:00, 1 min per action
        self.delta_t = dt.timedelta(seconds=60)    # deltat = 1 min
        self.orders_packet_idx = 0
        self.training_df = pd.read_csv("data/" + str(self.orders_packet_idx) + ".csv")[0:15]

        self.max_received_num = 3
        self.max_carrying_num = 3
        self.living_cost = -1
        self.reward_per_order = 10
        self.penalty_per_order = 9

        self.init_agent_state = (self.init_agent_pos, util.datetime_to_str(self.start_time), (), ())


    def getImmOrders(self, _str_time):
        """
        Return the list of immediate orders
        """
        curr_time_string = _str_time
        curr_df = self.training_df[self.training_df['created_time'] == curr_time_string]
        ImmOrders = tuple([tuple(x) for x in curr_df.values])
        return ImmOrders

    def getOneOrder_inTuple(self, _order_idx):
        return tuple(self.training_df.iloc[_order_idx].values.tolist())

    def getSuccessorStateandReward(self, _state, _action):
        """
        * state = (pos, str_time, (received), (carrying))
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
        nextState.append(list(_state[2]))
        for received_idx in _action[1]:
            nextState[2].append(self.getOneOrder_inTuple(received_idx))
        # Update received orders to carrying status
        nextState.append(list(_state[3]))
        if _state[0] in self.map.restaurants_poss:
            cur_res_idx = self.map.restaurants_poss.index(_state[0])
            cur_res_name = 'res_' + chr(ord('A') + cur_res_idx)
            for received_order in nextState[2]:
                # Limit on # of carrying orders
                if cur_res_name in received_order and len(nextState[3]) < self.max_carrying_num:
                    nextState[3].append(received_order)
                    nextState[2].remove(received_order)
        # Update carrying orders
        actual_reward = self.living_cost
        if _state[0] == self.map.des_pos:
            tot_reward = self.reward_per_order * len(nextState[3])
            tot_penalty = 0
            for order in nextState[3]:
                if dt_curr_time > util.getDueTime(order):
                    tot_penalty += self.penalty_per_order
            nextState[3] = []
            actual_reward = tot_reward - tot_penalty
        # Convert order lists to tuples
        nextState[2] = tuple(nextState[2])
        nextState[3] = tuple(nextState[3])
        return tuple(nextState), actual_reward

    def getFinalStateValue(self, _state):
        assert(len(self.getLegalActions(_state)) == 0)
        exceed_orders_num = 0
        for order in _state[2] + _state[3]:
            if util.str_to_datetime(_state[1]) >= util.getDueTime(order):
                exceed_orders_num += 1
        return -exceed_orders_num * self.penalty_per_order

    def getLegalActions(self, _state):
        """
        state = (pos, str_time, [received], [carrying])
        Get the legal actions for the agent:
        * Move east/west/south/north
        * Receive/Reject orders
        * Stay down and do nothing
        e.g. an action ('South', [0,1], [2]) means to go south, while receiving order 0&1 but rejecting 2
        """
        if util.str_to_datetime(_state[1]) >= self.end_time:
            return []
        
        successors = self.map.get_successor(_state[0])
        assert(len(successors)>0)
        # move_actions = ['Stay']
        move_actions = []
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
        # Decrease choices according to the orders in current state
        curr_received_num = len(_state[2])
        order_actions = [valid_action for valid_action in order_actions if curr_received_num + len(valid_action[0]) <= self.max_received_num]
        actions = []
        for move_action in move_actions:
            for order_action in order_actions:
                # order_action = ([received_idxs],[rejected_idxs])
                actions.append((move_action, order_action[0], order_action[1]))
        
        return actions

    def isTerminal(self, _state):
        return len(self.getLegalActions(_state)) == 0

    def trainWeights(self, numIter = 100):
        """
        Start from an innocent agent, repeat the delivery period for numIter times.
        Actions are chosen arbitrarily.
        """
        for iter_idx in range(numIter):
            state = self.init_agent_state
            while not self.isTerminal(state):
                # print(state)
                action = self.agent.getAction_byQvalues(state)
                nextState, reward = self.getSuccessorStateandReward(state, action)
                self.agent.update(state, action, nextState, reward)
                state = nextState
            if iter_idx % 10 == 0:
                print("iter", iter_idx)
                if iter_idx > 2:
                    self.testOneEpisode_byQvalues()
            # TODO: plot this figure

    def valueIter(self, numIter = 1000):
        # Fill all states
        states = []
        init_state = self.init_agent_state
        queue = util.Queue()
        queue.push(init_state)
        while not queue.isEmpty():
            if len(states) > 0 and math.log(len(states),10) - int(math.log(len(states),10)) == 0:
                print("states num = 10 ^ ", math.log(len(states),10))
                print(states[-1])
            state = queue.pop()
            states.append(state)
            for action in self.getLegalActions(state):
                successor, _ = self.getSuccessorStateandReward(state, action)
                if successor not in queue.list and successor not in states:
                    queue.push(successor)
        values = util.Counter()

        for iter_idx in range(numIter):
            if iter_idx % 50 == 0:
                print("iter ", iter_idx)
                self.testOneEpisode_byValues(values)
            new_values = util.Counter()
            for state in states:
                actions = self.getLegalActions(state)
                if len(actions) == 0:
                    new_values[state] = self.getFinalStateValue(state)
                else:
                    values_on_action = []
                    for action in actions:
                        successor, reward = self.getSuccessorStateandReward(state, action)
                        values_on_action.append(reward + self.agent.gamma * values[successor])
                    new_values[state] = max(values_on_action)
            values = new_values.copy()

    def policyIter_TDL(self, improve_times = 1000, epis_per_improve = 100):
        """
        Policy eval(TDL) and improve
        """
        for improve_idx in range(improve_times):
            values_pi = util.Counter()  # values under this policy
            for episode_idx in range(epis_per_improve):
                if episode_idx % 50 == 0:
                    print("improve times: ", improve_idx, "episode: ", episode_idx)
                state = self.init_agent_state
                # Run an episode
                while not self.isTerminal(state):
                    successor, reward = self.getSuccessorStateandReward(state, self.agent.getAction_byDict(state))
                    sample = reward + self.agent.gamma * values_pi[successor]
                    values_pi[state] = (1-self.agent.alpha) * values_pi[state] + self.agent.alpha * sample
                    state = successor
            # Improve policy
            new_policy = util.Counter()
            for state in values_pi:
                if self.isTerminal(state):
                    continue
                qvalues = []
                actions = self.getLegalActions(state)
                for action in actions:
                    successor, reward = self.getSuccessorStateandReward(state, action)
                    if successor in values_pi:
                        qvalues.append(reward + self.agent.gamma * values_pi[successor])
                    else:
                        qvalues.append(reward)
                max_qvalue = max(qvalues)
                best_actions_idx = [i for i in range(len(actions)) if qvalues[i] == max_qvalue]
                new_policy[state] = rd.choice(best_actions_idx)
            self.agent.policy = new_policy.copy()
            # See the outcome of current policy
            # if improve_idx % 100 == 0:
            self.testOneEpisode_byDict()

    def testOneEpisode_byQvalues(self):
        """
        Test the policy for one episode
        """
        score = 0
        state = self.init_agent_state
        actions = self.getLegalActions(state)
        while len(actions) != 0:
            action = self.agent.getPolicy_byQvalues(state)
            nextState, reward = self.getSuccessorStateandReward(state, action)
            score += reward
            state = nextState
            actions = self.getLegalActions(state)
        print("score = ", score)

    def testOneEpisode_byValues(self, _values):
        """
        Test the policy for one episode
        """
        score = 0
        state = self.init_agent_state
        actions = self.getLegalActions(state)
        while len(actions) != 0:
            action = self.agent.getPolicy_byValues(state, _values)
            nextState, reward = self.getSuccessorStateandReward(state, action)
            score += reward
            state = nextState
            actions = self.getLegalActions(state)
        print("score = ", score)

    def testOneEpisode_byDict(self):
        """
        Test the policy for one episode
        """
        score = 0
        state = self.init_agent_state
        while not self.isTerminal(state):
            action = self.agent.getPolicy_byDict(state)
            nextState, reward = self.getSuccessorStateandReward(state, action)
            score += reward
            state = nextState
            actions = self.getLegalActions(state)
        print("score = ", score)

if __name__ == "__main__":
    zhangjiang = World()
    # zhangjiang.valueIter()
    zhangjiang.trainWeights()
    # zhangjiang.policyIter_TDL()



        
        

        