# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    self.qValues = util.Counter()
    print ("ALPHA", self.alpha)
    print ("DISCOUNT", self.discount)
    print ("EXPLORATION", self.epsilon)

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    return self.qValues[(state, action)]


  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    possibleStateQValues = util.Counter()
    for action in self.getLegalActions(state):
    	possibleStateQValues[action] = self.getQValue(state, action)
    
    return possibleStateQValues[possibleStateQValues.argMax()]

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    possibleStateQValues = util.Counter()
    possibleActions = self.getLegalActions(state)
    if len(possibleActions) == 0:
    	return None
    
    for action in possibleActions:
    	possibleStateQValues[action] = self.getQValue(state, action)
    
    if possibleStateQValues.totalCount() == 0:
    	return random.choice(possibleActions)
    else:
    	return possibleStateQValues.argMax()

  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick Action
    legalActions = self.getLegalActions(state)
    action = None
    if len(legalActions) > 0:
      if util.flipCoin(self.epsilon):
    	  action = random.choice(legalActions)
      else:
			    action = self.getPolicy(state)

    return action

  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    print ("State: ", state, " , Action: ", action, " , NextState: ", nextState, " , Reward: ", reward)
    print ("QVALUE", self.getQValue(state, action))
    print ("VALUE", self.getValue(nextState))
    self.qValues[(state, action)] = self.getQValue(state, action) + self.alpha * (reward + self.discount * self.getValue(nextState) - self.getQValue(state, action))

class PacmanQAgent(QLearningAgent):
  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    self.index = 0  # This is always Pacman
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action


class ApproximateQAgent(PacmanQAgent):
  numActions = 0
  reward = 0
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)

    # You might want to initialize weights here.
    self.weights = util.Counter()

  def getQValue(self, state, action):
    qValue = 0.0
    features = self.featExtractor.getFeatures(state, action)
    for key in features.keys():
    	qValue += (self.weights[key] * features[key])
    return qValue

  def update(self, state, action, nextState, reward):
    self.numActions = self.numActions + 1
    self.reward = reward
    features = self.featExtractor.getFeatures(state, action)
    possibleStateQValues = []
    for act in self.getLegalActions(state):
      possibleStateQValues.append(self.getQValue(state, act))

    for key in features.keys():
      self.weights[key] += self.alpha * (reward + self.discount * ((1-self.epsilon)*self.getValue(nextState)+(self.epsilon/len(possibleStateQValues))*(sum(possibleStateQValues))) - self.getQValue(state, action)) * features[key]
      # print("key: {}, weights: {}".format(key, self.weights[key]))
    
  def final(self, state):
    file = open('stats.txt', 'a')
    gameResult = 0
    if(state.isWin()):
      gameResult = 1

    actions = str(self.numActions) + ',' + str(state.data.score) + '\n'
    file.write(actions)
    file.close()
    self.numActions = 0
    PacmanQAgent.final(self, state)

    if self.episodesSoFar == self.numTraining:
      pass
