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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
        # Generate the successor game state
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        
        # Get the new position of Pac-Man
        newPos = successorGameState.getPacmanPosition()
        
        # Get the new food grid and ghost states
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        
        # Base score from the successor state
        score = successorGameState.getScore()
        
        # Food heuristic: Encourage moving closer to food
        foodList = newFood.asList()
        if foodList:
            minFoodDist = min(manhattanDistance(newPos, food) for food in foodList)
            score += 10 / minFoodDist  # Prioritize food (higher weight)
    
        # Ghost heuristic: Avoid getting too close to ghosts
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            ghostDist = manhattanDistance(newPos, ghostPos)
            if ghostDist <= 1:  # Immediate danger
                score -= 1000  # Huge penalty for running into a ghost
            else:
                score -= 5 / ghostDist  # Smaller penalty for getting close
    
        return score


def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
        # Determine the max value of the current agent (Pacman)
        def maxValue(state, depth, agentIndex):

            #get list of actions
            actions = state.getLegalActions(agentIndex)

            if not actions:
                return self.evaluationFunction(state) #return the score if no actions are available
            
            maxVal = float("-inf")
            maxAction = None

            for action in actions:
                successor = state.generateSuccessor(agentIndex, action) #generate the successor state to make a plan
                val = outcome(successor, depth, agentIndex + 1) #Recursively call the value function to get the value of the successor ghost statea
                if val > maxVal:
                    maxVal = val
                    maxAction = action
            
            if depth == 0:
                return maxAction
            return maxVal
        
        # Determine the min value of the current agent (Ghosts)
        def minValue(state, depth, agentIndex):
            actions = state.getLegalActions(agentIndex)

            # If no actions are available, return the score
            if not actions:
                return self.evaluationFunction(state)
            
            minVal = float("inf")
            nextAgent = agentIndex + 1
            # If the next agent is the last agent, increment the depth and reset the agent index
            if nextAgent == state.getNumAgents():
                nextAgent = 0
                depth += 1

            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                val = outcome(successor, depth, nextAgent)
                minVal = min(minVal, val)
            return minVal
      
      
        # Compare the mins and maxes of the agents to determine the best action
        def outcome(state, depth, agentIndex):
            # If the depth is equal to the max depth, or the game is won or lost, return the score
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            # If the agent index is 0, return the max value for Mr. Pacman
            if agentIndex == 0:
                return maxValue(state, depth, agentIndex) 
            # Otherwise, return the min value for Ghosts
            else:
                return minValue(state, depth, agentIndex)


        return maxValue(gameState, 0, 0)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # Most of this is reused from the MinimaxAgent class
        def maxValue(state, depth, agentIndex, alpha, beta):

            #get list of actions
            actions = state.getLegalActions(agentIndex)

            if not actions:
                return self.evaluationFunction(state) #return the score if no actions are available
            
            maxVal = float("-inf")
            maxAction = None

            for action in actions:
                successor = state.generateSuccessor(agentIndex, action) #generate the successor state to make a plan
                val = outcome(successor, depth, 1, alpha, beta) #Recursively call the value function to get the value of the successor ghost statea
                if val > maxVal:
                    maxVal = val
                    maxAction = action
                if maxVal > beta:
                    return maxVal #Pruned
                alpha = max(alpha, maxVal)
            
            if depth == 0: # Return at the root node
                return maxAction
            
            return maxVal
        
        # Determine the min value of the current agent (Ghosts)
        def minValue(state, depth, agentIndex, alpha, beta):
            actions = state.getLegalActions(agentIndex)

            # If no actions are available, return the score
            if not actions:
                return self.evaluationFunction(state)
            
            minVal = float("inf")
            nextAgent = agentIndex + 1
            # If the next agent is the last agent, increment the depth and reset the agent index
            if nextAgent == state.getNumAgents():
                nextAgent = 0
                depth += 1

            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                val = outcome(successor, depth, nextAgent, alpha, beta)
                minVal = min(minVal, val)
                if minVal < alpha:
                    return minVal
                beta = min(beta, minVal)
            return minVal
        
        # Compare the mins and maxes of the agents to determine the best action
        def outcome(state, depth, agentIndex, alpha, beta):
            # If the depth is equal to the max depth, or the game is won or lost, return the score
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            # If the agent index is 0, return the max value for Mr. Pacman
            if agentIndex == 0:
                return maxValue(state, depth, agentIndex, alpha, beta) 
            # Otherwise, return the min value for Ghosts
            else:
                return minValue(state, depth, agentIndex, alpha, beta)

        return outcome(gameState, 0, 0, float("-inf"), float("inf"))

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(state, depth, agentIndex):
            """
            Recursively calculates the best move using the Expectimax decision tree.
            
            - If depth is reached or the game is won/lost, return the evaluation function score.
            - If the agent is Pacman (index 0), return the best possible score (maximization).
            - If the agent is a ghost, return the expected score (average of all possible outcomes).
            """

            # Base case: If depth limit is reached or game is over, evaluate the state.
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # Get all legal actions for the current agent.
            actions = state.getLegalActions(agentIndex)
            if not actions:  # If no actions available, return the evaluation score.
                return self.evaluationFunction(state)

            # Generate successor states for each possible action.
            successors = [state.generateSuccessor(agentIndex, action) for action in actions]

            # Pacman (Maximizing Player)
            if agentIndex == 0:
                # If at root node (depth 0), return the action corresponding to the max value.
                if depth == 0:
                    return max((expectimax(successor, depth, 1), action) for successor, action in zip(successors, actions))[1]
                # Otherwise, return the max value found.
                return max(expectimax(successor, depth, 1) for successor in successors)

            # Ghosts (Chance Nodes - Average Value)
            else:
                nextAgent = agentIndex + 1
                if nextAgent == state.getNumAgents():  # If last ghost, increase depth and reset agent index.
                    nextAgent = 0
                    depth += 1
                # Return the average (expected value) of all possible successor states.
                return sum(expectimax(successor, depth, nextAgent) for successor in successors) / len(successors)

        # Call expectimax on the initial game state and return the best action.
        return expectimax(gameState, 0, 0)
        
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # Get Pacman's position
    pacmanPos = currentGameState.getPacmanPosition()

    # Get food, ghosts, and power pellet locations
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsuleList = currentGameState.getCapsules()

    # Start with current game score
    score = currentGameState.getScore()

    # Encourage Pacman to move toward closest food pellet
    if foodList:
        minFoodDist = min(manhattanDistance(pacmanPos, food) for food in foodList)
        score += 10 / minFoodDist # Higher weigh prioritizes food collection

    # Discourage Pacman from getting close to ghosts
    for ghost in ghostStates:
        ghostDist = manhattanDistance(pacmanPos, ghost.getPosition())

        if ghost.scaredTimer == 0: # If ghost is active
            if ghostDist <= 1: # Immediate danger
                score -= 1000
            else:
                score -= 5 / ghostDist # Small penalty for getting close to any ghost

        else: # If ghost scared
            score += 50 / ghostDist # Incentive to move towards scared ghost

    #Encourage Pacman to eat power pellet
    if capsuleList:
        minCapsuleDist = min(manhattanDistance(pacmanPos, capsule) for capsule in capsuleList)
        score += 20 / minCapsuleDist # Incentive picking up power pellet

    # Reward Pacman for having fewer food pellets left on board
    score -= len(foodList) * 5 # Small penalty for remaining food pellets

    return score
    
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
