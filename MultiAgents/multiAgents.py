# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from email.header import make_header
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodList = newFood.asList()
        minFoodDist = -float('inf')
        for food in foodList:
            dist = manhattanDistance(newPos, food)
            if minFoodDist >= dist or minFoodDist == -float('inf'):
                minFoodDist = dist

        ghostDist = 1
        numGhostSurround = 0
        for ghost in successorGameState.getGhostPositions():
            dist = manhattanDistance(newPos, ghost)
            ghostDist += dist
            if dist <= 1:
                numGhostSurround += 1

        return successorGameState.getScore() + (1 / float(minFoodDist)) - (1 / float(ghostDist)) - numGhostSurround


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def maxLvl(gameState,depth):
            currDepth = depth + 1
            if gameState.isWin() or gameState.isLose() or currDepth==self.depth:
                return self.evaluationFunction(gameState)
            maxvalue = -float('inf')
            for action in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, action)
                maxvalue = max(maxvalue, minLvl(successor,currDepth,1))
            return maxvalue
        
        def minLvl(gameState,depth, agentIndex):
            minvalue = float('inf')
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            for action in gameState.getLegalActions(agentIndex):
                successor= gameState.generateSuccessor(agentIndex,action)
                if agentIndex == (gameState.getNumAgents() - 1):
                    minvalue = min(minvalue,maxLvl(successor,depth))
                else:
                    minvalue = min(minvalue,minLvl(successor,depth,agentIndex+1))
            return minvalue
        
        #Root level action.
        currScore = -float('inf')
        action = 0
        for a in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0,a)
            score = minLvl(nextState,0,1)
            if score > currScore or currScore == -float('inf'):
                action = a
                currScore = score
        return action
        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(gameState, depth, agentIndex, a, b):  # maximizer function
            v = float("-inf")
            for newState in gameState.getLegalActions(agentIndex):
                v = max(v, alphabetaprune(gameState.generateSuccessor(agentIndex, newState), depth, 1, a, b))
                if v > b:
                    return v
                a = max(a, v)
            return v

        def min_value(gameState, depth, agentIndex, a, b):  # minimizer function
            v = float("inf")
            nextAgent = agentIndex + 1  # calculate the next agent and increase depth accordingly.
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:
                depth += 1

            for action in gameState.getLegalActions(agentIndex):
                v = min(v, alphabetaprune(gameState.generateSuccessor(agentIndex, action), depth, nextAgent, a, b))
                if v < a:
                    return v
                b = min(b, v)
            return v

        def alphabetaprune(gameState, depth, agentIndex, a, b):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:  # return the utility in case the defined depth is reached or the game is won/lost.
                return self.evaluationFunction(gameState)

            if agentIndex == 0:  # maximize for pacman
                return max_value(gameState, depth, agentIndex, a, b)
            else:  # minimize for ghosts
                return min_value(gameState, depth, agentIndex, a, b)

        v = float("-inf")
        action = ''
        a = float("-inf")
        b = float("inf")
        for x in gameState.getLegalActions(0):
            value = alphabetaprune(gameState.generateSuccessor(0, x), 0, 1, a, b)
            if value > v:
                v = value
                action = x
            if v > b:
                return v
            a = max(a, v)

        return action

        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(gameState, depth, agentIndex):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agentIndex == 0:
                possibles = []
                for action in gameState.getLegalActions(agentIndex):
                    possibles.append(expectimax(gameState.generateSuccessor(agentIndex, action), depth, 1))
                return max(possibles)
            else:
                nextAgent = agentIndex + 1
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                if nextAgent == 0:
                    depth += 1
                possibles = []
                for action in gameState.getLegalActions(agentIndex):
                    possibles.append(expectimax(gameState.generateSuccessor(agentIndex, action), depth, nextAgent))
                return float(sum(possibles) / float(len(gameState.getLegalActions(agentIndex))))

        maximum = float("-inf")
        action = ''
        for a in gameState.getLegalActions(0):
            v = expectimax(gameState.generateSuccessor(0, a), 0, 1)
            if v > maximum or maximum == float("-inf"):
                maximum = v
                action = a

        return action

        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    minFoodDist = -float('inf')
    for food in foodList:
        dist = manhattanDistance(newPos, food)
        if minFoodDist >= dist or minFoodDist == -float('inf'):
            minFoodDist = dist

    ghostDist = 1
    numGhostSurround = 0
    for ghost in currentGameState.getGhostPositions():
        dist = util.manhattanDistance(newPos, ghost)
        ghostDist += dist
        if dist <= 1:
            numGhostSurround += 1

    capsule = currentGameState.getCapsules()
    numCapsule = len(capsule)

    return currentGameState.getScore() + (1 / float(minFoodDist)) - (1 / float(ghostDist)) - numGhostSurround - numCapsule

    # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
