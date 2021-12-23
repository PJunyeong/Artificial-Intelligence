# search.py
# ---------
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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
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

    Stack = util.Stack() # DFS implemented using stack.
    Visited = set() # check if CurNode is already visited using set. 
    StartNode = problem.getStartState() ; FromState = tuple() ; ToDir = str()
    Stack.push((StartNode, FromState, ToDir)) 
    # let State be (position, direction). using this state's info make a nested structure, from which a path is taken.
    # pushed tuple composed of (current position, previously pushed tuple, next direction)
    # since there's no previous node or directional info, let FromState and ToDir be None.
    while ~Stack.isEmpty():
        CurState = Stack.pop() 
        # Since DFS, go deeper until there's no other children. if no goal, just return and restart using remaining nodes info in stack.
        CurNode = CurState[0]
        # Each popped tuple has an info about where to be, come from and go: note that each tuple connected using FromState while completing the path.
        if problem.isGoalState(CurNode): # goal node taken and make a path by connecting ToDir (i.e. from goal to start position)
            DFS_Path = list() 
            while CurState[1]:
                DFS_Path.append(CurState[2])
                CurState = CurState[1]
            return DFS_Path[::-1] # as recorded from the goal, make its order reversed and return. 

        # if CurNode is goal, just return CurPath. Being chosen and updated from popped CurState, it is destined to be completed.
        if CurNode in Visited: continue # skip visited node.
        else:
            Visited.add(CurNode) # check current position as visited.
            Children = problem.getSuccessors(CurNode) # 
            for Child in Children: # bring each Child composed of (State, Direction)
                NextNode = Child[0]
                NextPath = Child[1]
                Stack.push((NextNode, CurState, NextPath)) # in order to memorize the previous nodes' condition, CurState is added as FromState variable.
    return list() # if there is no possible path, just return path as blank.
    util.raiseNotDefined()

def iterativeDeepeningdepthFirstSearch(problem):
    "*** MY CODE HERE : source code for 3rd discussion of q2***"
    Stack = util.Stack()
    MaxDepth = 1 # iterative deepening DFS as MaxDepth (or Limit) set = 1 initially.

    while True: 
        Visited = set()
        StartNode = problem.getStartState() ; FromState = tuple() ; ToDir = str() ; CurDepth = 0
        Stack.push((StartNode, FromState, ToDir, CurDepth)) 
        # CurDepth added to show in which depth CurNode lies in. if larger than current MaxDepth, do not go further. 
        CurState = Stack.pop()
        CurNode = CurState[0]
        CurDepth = CurState[3]
        Visited.add(CurNode)

        while (problem.isGoalState(CurNode) == False): # if goal found or stack empty, escape this loop:
            Children = problem.getSuccessors(CurNode)
            for Child in Children: 
                NextNode = Child[0]
                NextPath = Child[1]
                NextDepth = CurDepth + 1 # while searching, NextNode's depth computed and checked using MaxDepth.
                if NextNode in Visited: continue
                if (NextDepth <= MaxDepth):
                    Stack.push((NextNode, CurState, NextPath, NextDepth))
                    Visited.add(NextNode)
            if Stack.isEmpty(): break # if there's no goal in searched nodes until Max_Depth, go outside this loop:  
            CurState = Stack.pop()
            CurNode = CurState[0]
            CurDepth = CurState[3]
            Visited.add(CurNode)

        if problem.isGoalState(CurNode): # if found, connect directions and complete path using backtracking as in other searching.
            IDS_Path = list() 
            while CurState[1]:
                IDS_Path.append(CurState[2])
                CurState = CurState[1]
            return IDS_Path[::-1]

        MaxDepth += 1 # if nested while loop ended and goal not found, then restart searching with MaxDepth increased by 1.

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first.""" # Comments on those parts same in the DFS comments would be skipped.
    "*** YOUR CODE HERE ***"
    Queue = util.Queue() # BFS implemented using stack.
    Visited = set() 
    StartNode = problem.getStartState() ; FromState = tuple() ; ToDir = str()
    Queue.push((StartNode, FromState, ToDir)) 
    while ~Queue.isEmpty():
        CurState = Queue.pop() # Since BFS, visit all of unvisited nodes at the same level.
        CurNode = CurState[0]
        if problem.isGoalState(CurNode): 
            BFS_Path = list() 
            while CurState[1]:
                BFS_Path.append(CurState[2])
                CurState = CurState[1]
            print(BFS_Path)
            return BFS_Path[::-1] 

        if CurNode in Visited: continue
        else:
            Visited.add(CurNode)
            Children = problem.getSuccessors(CurNode) 
            for Child in Children: 
                NextNode = Child[0]
                NextPath = Child[1]
                Queue.push((NextNode, CurState, NextPath)) 
    return list() 
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    PQueue = util.PriorityQueue() # UCS implemented using priority queue as node with min cost selected.
    Visited = set() 
    StartNode = problem.getStartState() ; FromState = tuple() ; ToDir = str() ; ToCost = 0
    PQueue.push((StartNode, FromState, ToDir, ToCost), ToCost) 
    # Since UCS, cost would be considered as tuple element. 
    # ToCost referred inside tuple for sum cost to that node and outside for pq key.
    while ~PQueue.isEmpty():
        CurState = PQueue.pop() # Since UFS, use PQ and visit the node which has a highest pq key (=min cost)
        CurNode = CurState[0]
        ToCost = CurState[3]
        if problem.isGoalState(CurNode): 
            UCS_Path = list() 
            while CurState[1]:
                UCS_Path.append(CurState[2])
                CurState = CurState[1]
            return UCS_Path[::-1] # same in terms of path connecting as implemented in DFS or BFS.

        if CurNode in Visited: continue
        else:
            Visited.add(CurNode)
            Children = problem.getSuccessors(CurNode) 
            for Child in Children: 
                NextNode = Child[0]
                NextPath = Child[1]
                NextCost = ToCost + Child[2] # to compute cost next possible routes, just sum up.
                PQueue.push((NextNode, CurState, NextPath, NextCost), NextCost) 
    return list() 
    util.raiseNotDefined()    

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    PQueue = util.PriorityQueue() # A* implemented using priority queue using f(n) as its pq key.
    Visited = set() 
    StartNode = problem.getStartState() ; FromState = tuple() ; ToDir = str() ; ToCost = 0
    PQueue.push((StartNode, FromState, ToDir, ToCost), heuristic(StartNode, problem) + ToCost) 
    # Since A*, f(n) = h(n) + g(n) used. h(n) for returned val from Heuristic func and g(n) from its cost.
    # ToCost referred inside tuple for sum cost to that node and outside for pq key.
    while ~PQueue.isEmpty():
        CurState = PQueue.pop() # Since A*, use PQ and visit the node which has a highest pq key (=f(n))
        CurNode = CurState[0]
        ToCost = CurState[3]
        if problem.isGoalState(CurNode): 
            Astar_Path = list() 
            while CurState[1]:
                Astar_Path.append(CurState[2])
                CurState = CurState[1]
            return Astar_Path[::-1] # same in terms of path connecting as implemented in DFS or BFS.

        if CurNode in Visited: continue
        else:
            Visited.add(CurNode)
            Children = problem.getSuccessors(CurNode) 
            
            for Child in Children: 
                NextNode = Child[0]
                NextPath = Child[1]
                NextCost = ToCost + Child[2] # to compute cost next possible routes, just sum up.
                PQueue.push((NextNode, CurState, NextPath, NextCost), heuristic(NextNode, problem) + NextCost)
                # in order to compute f(n) - PQ key val, push sum of h(n) + g(n) into PQ. 
    return list() 
    util.raiseNotDefined()    


# Abbreviations
bfs = breadthFirstSearch
ids = iterativeDeepeningdepthFirstSearch # added ids for 3rd discussion in q2. please check submitted report.
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
