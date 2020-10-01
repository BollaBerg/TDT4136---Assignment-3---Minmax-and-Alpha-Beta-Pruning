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
        return successorGameState.getScore()

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

    def max_value(self, gameState, depth : int) -> int:
        """Returns the max value of all possible actions from gameState"""
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        best_score = float("-inf")
        actions = gameState.getLegalActions(self.index)

        for action in actions:
            # Get the next gameState, which is current gameState + action
            successor_state = gameState.generateSuccessor(self.index, action)
            # Get the resulting score for the current action and compare it to 
            # current best score. Keep the best
            best_score = max(best_score, self.min_value(successor_state, depth, 1))
        
        return best_score

    def min_value(self, gameState, depth : int, index : int) -> int:
        """Returns the min value of all possible actions from gameState"""
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        lowest_score = float("inf")
        actions = gameState.getLegalActions(index)

        for action in actions:
            # Get the next gameState, which is current gameState + action
            successor_state = gameState.generateSuccessor(index, action)
            
            if index == self.number_of_agents - 1:
                # Last of the minimizing agents (ghosts)
                # --> Next step is a maximizing agent (PacMan)
                value = self.max_value(successor_state, depth + 1)
            else:
                # Not last of the minimizing agents (ghosts)
                # --> Next step is a minimizing agent (next ghost)
                value = self.min_value(successor_state, depth, index + 1)
            
            # Compare the value (score) of the current scenario to the current
            # lowest score. Keep the lowest
            lowest_score = min(lowest_score, value)

        return lowest_score
        

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
        self.number_of_agents = gameState.getNumAgents()
        actions = gameState.getLegalActions(self.index)

        best_score = float("-inf")
        best_action = None

        for action in actions:
            # Get next state (resulting from action)
            successor_state = gameState.generateSuccessor(self.index, action)
            # Get the resulting score for the current action
            score = self.min_value(successor_state, 0, 1)

            # We want the action that gives the best score
            if score > best_score:
                best_action = action
                best_score = score

        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def max_value(self, gameState, depth : int, alpha : float, beta : float) -> float:
        """Returns the max value of all possible actions from gameState"""
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        best_score = float("-inf")
        actions = gameState.getLegalActions(self.index)

        for action in actions:
            # Get the next gameState, which is current gameState + action
            successor_state = gameState.generateSuccessor(self.index, action)
            # Get the resulting score for the current action and compare it to 
            # current best score. Keep the best
            best_score = max(best_score, self.min_value(successor_state, depth, 1, alpha, beta))

            # Alpha-beta implementation:
            # if best_score is higher than beta, that means that the best score of the maximizing node
            # will be higher than whatever it will be compared to in the minimizing parent of this.
            # No matter what values would've been checked after this, they could only increase the score.
            # If only the lowest score (of beta and best_score) will be used anyways, then there's no
            # need to check any other actions --> No result will bring best_score down below beta anyways,
            # so the other value (beta) will be chosen
            if best_score > beta: return best_score

            # Set alpha (the current highest value for this branch --> whatever will be quickly selected
            # for the previous maximizing-node) to be the highest of alpha and best_score
            # Meaning: If this best_score is higher than what was previously going to be selected --> update
            # what score is going to be selected (and thus compared to in self.min_value)
            alpha = max(alpha, best_score)

        
        return best_score

    def min_value(self, gameState, depth : int, index : int, alpha : float, beta : float) -> float:
        """Returns the min value of all possible actions from gameState"""
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        lowest_score = float("inf")
        actions = gameState.getLegalActions(index)

        for action in actions:
            # Get the next gameState, which is current gameState + action
            successor_state = gameState.generateSuccessor(index, action)
            
            if index == self.number_of_agents - 1:
                # Last of the minimizing agents (ghosts)
                # --> Next step is a maximizing agent (PacMan)
                value = self.max_value(successor_state, depth + 1, alpha, beta)
            else:
                # Not last of the minimizing agents (ghosts)
                # --> Next step is a minimizing agent (next ghost)
                value = self.min_value(successor_state, depth, index + 1, alpha, beta)
            
            # Compare the value (score) of the current scenario to the current
            # lowest score. Keep the lowest
            lowest_score = min(lowest_score, value)

            # Alpha-beta implementation:
            # if lowest_score is lower than alpha, that means that the lowest score of the minimizing node
            # will be lower than whatever it will be compared to in the maximizing parent of this.
            # No matter what values would've been checked after this, they could only lower the score.
            # If only the highest score (of alpha and lowest_score) will be used anyways, then there's no
            # need to check any other actions --> No result will increase lowest_score above alpha anyways,
            # so the other value (alpha) will be chosen
            if lowest_score < alpha: return lowest_score

            # Set beta (the current lowest value for this branch --> whatever will be selected
            # for the previous minimizing-node) to be the lowest of beta and lowest_score
            # Meaning: If this lowest_score is lower than what was previously going to be selected --> update
            # what score is going to be selected (and thus compared to in self.max_value)
            beta = min(beta, lowest_score)

        return lowest_score
        

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Note: The actual code here is pretty much the same as for a max_value node.
        This makes sense, as the first step actually is to maximize the value of
        the first action taken.
        The only reason this is not just a call to max_value() is that this needs
        to return an action as well. While I could refactor the code so that max_value()
        actually returns an action, it is easier this way. Especially as this code only
        needs to solve the given problems, there's no need to generalize further.
        """
        self.number_of_agents = gameState.getNumAgents()
        actions = gameState.getLegalActions(self.index)

        best_score = float("-inf")
        best_action = None

        alpha = float("-inf")
        beta = float("inf")

        for action in actions:
            successor_state = gameState.generateSuccessor(self.index, action)
            score = self.min_value(successor_state, 0, 1, alpha, beta)

            if score > best_score:
                best_action = action
                best_score = score

            # Because this is practically a max_value node (it actually is. The only
            # reason to keep this as a separate thing is because it needs to return
            # an action):
            # update alpha as you would for a normal max_value node
            alpha = max(alpha, score)

        return best_action

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
