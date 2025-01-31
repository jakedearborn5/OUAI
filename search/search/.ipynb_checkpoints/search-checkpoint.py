# search.py
# ---------
# Source code for CS4013/5013--P1 
# ------------------------------------------------------------------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    from util import Stack #Import Stack data structure

    #Initialize the stack and a set to track visited nodes
    fringe = Stack()
    visited = set()

    #Push the start state onto stack
    fringe.push((problem.getStartState(), []))

    while not fringe.isEmpty():
        state, path = fringe.pop() #Get current state and path

        if state in visited:
            continue #Skip visited states
    
        visited.add(state) #Mark state as visited
    
        if problem.isGoalState(state):
            return path #return path once goal is found
    
        for successor, action, _ in problem.getSuccessors(state):
            if successor not in visited:
                #Push successor and updated path onto stack
                fringe.push((successor, path + [action]))

    return [] #Return empty if no solution
    
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    from util import Queue #Import Queue data structure

    #Initialize queue and set to track visited nodes
    fringe = Queue()
    visited = set()

    #Enqueue start state
    fringe.push((problem.getStartState(), []))

    while not fringe.isEmpty():
        state, path = fringe.pop() #Get current state and path

        if state in visited:
            continue #Skip visited states

        visited.add(state) #Mark state as visited

        if problem.isGoalState(state):
            return path #Return path when goal is found

        for successor, action, _ in problem.getSuccessors(state):
            if successor not in visited:
                #Enqueue successor and updated path
                fringe.push((successor, path + [action]))

    return [] #Return empty if no solution

    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    from util import PriorityQueue #Import Prioirty Queue data structure

    #Initialize queue and set to track visited nodes
    fringe = PriorityQueue()
    visited = set()

    #Enqueue start state
    fringe.push((problem.getStartState(), []), 0)

    while not fringe.isEmpty():
        state, path = fringe.pop() #Get current state and path

        if state in visited:
            continue #Skip visited states
    
        visited.add(state) #Mark state as visited
    
        if problem.isGoalState(state):
            return path #Return path when goal is found
    
        for successor, action, _ in problem.getSuccessors(state):
            if successor not in visited:
                #Enqueue successor and updated path
                fringe.push((successor, path + [action]), problem.getCostOfActions(path + [action]))

    return [] #Return empty if no solution
    
    util.raiseNotDefined()

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    from util import PriorityQueue

    fringe = PriorityQueue()  # Priority queue for A*
    visited = {}  # Tracks the lowest known cost for each state (using default set stucture bc of typing errors)

    start_state = problem.getStartState()
    initial_heuristic = heuristic(start_state, problem)

    # Push the start state onto the fringe with f(n) = g(n) + h(n)
    fringe.push((start_state, [], 0), 0 + initial_heuristic)  # (state, path, g(n)), priority = f(n)

    while not fringe.isEmpty():
        state, path, g = fringe.pop()

        # Skip if we've already found a cheaper path to this state
        if state in visited and visited[state] <= g:
            continue

        # Mark the state as visited with its current cost
        visited[state] = g

        # Check if this is the goal state
        if problem.isGoalState(state):
            return path

        for successor, action, stepCost in problem.getSuccessors(state):
            next_cost = g + stepCost

            # Skip if cheaper path found to this successor
            if successor in visited and visited[successor] <= next_cost:
                continue

            # Calculate the heuristic for the successor
            successor_heuristic = heuristic(successor, problem)

            f = next_cost + successor_heuristic

           # Push new successor onto the fringe with f(n) = g(n)(next_cost) + h(n)(successor_heuristic)
            fringe.push((successor, path + [action], next_cost), f)


    return []  # No solution found
    
    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
