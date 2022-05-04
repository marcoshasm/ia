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


from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    #return currentGameState.getScore()
    return betterEvaluationFunction(currentGameState)

def mindist(gameState, depth):
    position = gameState.getPacmanPosition()
    adj = []
    adj.append({"x": position[0], "y": position[1], "depth": 0})
    food = gameState.getFood()
    walls = gameState.getWalls()
    width = len(list(walls))
    height = len(walls[0])
    i = 0

    while True:
        if (adj[i]["depth"] > depth):
            break
        if (food[adj[i]["x"]][adj[i]["y"]]):
            return adj[i]["depth"]
        if (adj[i]["x"] + 1 < width and not walls[adj[i]["x"] + 1][adj[i]["y"]]):
            adj.append({"x": adj[i]["x"] + 1, "y": adj[i]["y"], "depth": adj[i]["depth"] + 1})
        if (adj[i]["x"] - 1 > 0 and not walls[adj[i]["x"] - 1][adj[i]["y"]]):
            adj.append({"x": adj[i]["x"] - 1, "y": adj[i]["y"], "depth": adj[i]["depth"] + 1})
        if (adj[i]["y"] + 1 < height and not walls[adj[i]["x"]][adj[i]["y"] + 1]):
            adj.append({"x": adj[i]["x"], "y": adj[i]["y"] + 1, "depth": adj[i]["depth"] + 1})
        if (adj[i]["y"] - 1 > 0 and not walls[adj[i]["x"]][adj[i]["y"] - 1]):
            adj.append({"x": adj[i]["x"], "y": adj[i]["y"] - 1, "depth": adj[i]["depth"] + 1})
        i += 1
    return depth + 1


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

    def minimax(self, gameState, depth, agent):
        bestMove = ''
        if agent == gameState.getNumAgents():
            agent = 0
            depth += 1

        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), bestMove

        if agent == 0:
            bestValue = -math.inf
            for action in gameState.getLegalActions(agent):
                val, move = self.minimax(gameState.generateSuccessor(agent, action), depth, agent + 1)
                if val > bestValue:
                    bestMove = action
                    bestValue = val
        else:
            bestValue = math.inf
            for action in gameState.getLegalActions(agent):
                val, move = self.minimax(gameState.generateSuccessor(agent, action), depth, agent + 1)
                if val < bestValue:
                    bestMove = action
                    bestValue = val

        return bestValue, bestMove

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        val, move = self.minimax(gameState, 0, 0)

        return move
        
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphabeta(self, gameState, depth, a, b, agent):
        bestMove = ''
        if agent == gameState.getNumAgents():
            agent = 0
            depth += 1

        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), bestMove

        if agent == 0:
            bestValue = -math.inf
            for action in gameState.getLegalActions(agent):
                val, move = self.alphabeta(gameState.generateSuccessor(agent, action), depth, a, b, agent + 1)
                if val > bestValue:
                    bestMove = action
                    bestValue = val
                a = max(a, bestValue)
                if b <= a:
                    break
        else:
            bestValue = math.inf
            for action in gameState.getLegalActions(agent):
                val, move = self.alphabeta(gameState.generateSuccessor(agent, action), depth, a, b, agent + 1)
                if val < bestValue:
                    bestMove = action
                    bestValue = val
                b = min(b, bestValue)
                if b < a:
                    break

        return bestValue, bestMove

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        val, move = self.alphabeta(gameState, 0, -math.inf, math.inf, 0)

        return move

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimax(self, gameState, depth, agent):
        bestMove = ''
        if agent == gameState.getNumAgents():
            agent = 0
            depth += 1

        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), bestMove

        if agent == 0:
            bestValue = -math.inf
            for action in gameState.getLegalActions(agent):
                val, move = self.expectimax(gameState.generateSuccessor(agent, action), depth, agent + 1)
                if val > bestValue:
                    bestMove = action
                    bestValue = val
        else:
            bestValue = 0
            for action in gameState.getLegalActions(agent):
                p = 1 / len(gameState.getLegalActions(agent))
                val, move = self.expectimax(gameState.generateSuccessor(agent, action), depth, agent + 1)
                bestValue += p * val
        
        return bestValue, bestMove

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        val, move = self.expectimax(gameState, 0, 0)

        return move

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()

    # achar distância minima para o ponto mais próximo
    score -= mindist(currentGameState, 5)
    return score
    util.raiseNotDefined()    

# Abbreviation
better = betterEvaluationFunction
