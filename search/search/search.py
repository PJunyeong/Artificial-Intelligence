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
    stack = util.Stack()
    stack.push([problem.getStartState(), []])
    visited = set()
    # 스택 사용 DFS. (현재 위치, 현재 경로) 확인

    while stack:
        cur_node, cur_path = stack.pop()
        if problem.isGoalState(cur_node): break
        if cur_node not in visited: visited.add(cur_node)
        # visited 집합으로 중복 방문 방지
        successors = problem.getSuccessors(cur_node)
        for successor in successors:
            next_node, next_dir, next_cost = successor
            # next_cost는 DFS, BFS에서 사용 X
            if next_node in visited: continue
            next_path = cur_path + [next_dir]
            # 다음 depth의 노드로 이어지는 '가능한' 경로 next_path
            stack.push([next_node, next_path])
    return cur_path
    # util.raiseNotDefined()

def iterativeDeepeningDepthFirstSearch(problem):
    # Q3의 IDS 풀이를 위한 소스 코드
    max_depth = 1
    # 최대 깊이 설정. 시작 깊이는 1.

    while True:
        # 최적 경로를 찾을 떄까지 깊이를 1씩 증가시켜 DFS.
        stack = util.Stack()
        stack.push([problem.getStartState(), [], 1])
        visited = set()
        visited.add(problem.getStartState())

        while not stack.isEmpty():
            cur_node, cur_path, cur_depth = stack.pop()
            if problem.isGoalState(cur_node): break

            successors = problem.getSuccessors(cur_node)
            for successor in successors:
                next_node, next_dir, next_cost = successor
                if next_node in visited or cur_depth + 1 > max_depth: continue
                # 다음 깊이가 탐색 가능하면 스택에 push.
                visited.add(next_node)
                next_path = cur_path + [next_dir]
                stack.push([next_node, next_path, cur_depth + 1])
        if problem.isGoalState(cur_node): return cur_path
        else: max_depth += 1
        # max_depth는 일반적으로 1씩 커질 때 optimal이 보장된다.


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    queue = util.Queue()
    queue.push([problem.getStartState(), []])
    visited = set()
    visited.add(problem.getStartState())
    # 큐 사용 BFS. (현재 위치, 현재 경로) 확인
    # queue에 '처음' 방문한 노드는 visited 체크

    while queue:
        cur_node, cur_path = queue.pop()
        if problem.isGoalState(cur_node): break
        successors = problem.getSuccessors(cur_node)
        for successor in successors:
            next_node, next_dir, next_cost = successor
            # child_cost는 DFS, BFS에서 사용 X
            if next_node not in visited:
                visited.add(next_node)
                # 미방문 노드라면, 방문하면서 visited로 체크
                next_path = cur_path + [next_dir]
                # 다음 depth의 노드로 이어지는 '가능한' 경로 next_path
                queue.push([next_node, next_path])

    return cur_path
    # util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    pq = util.PriorityQueue()
    pq.push([problem.getStartState(), [], 0], 0)
    visited = set()
    # 우선순위 큐 사용 UCS. (현재 위치, 현재 경로, 전체 비용) 확인.

    while pq:
        cur_node, cur_path, cur_cost = pq.pop()
        if problem.isGoalState(cur_node): break
        if cur_node in visited: continue
        # visited 집합으로 중복 방문 방지
        visited.add(cur_node)
        successors = problem.getSuccessors(cur_node)
        for successor in successors:
            next_node, next_dir, next_cost = successor
            next_path = cur_path + [next_dir]
            # 현재 노드까지 온 비용 + next_node 비용. next_path는 DFS, BFS와 동일
            pq.push([next_node, next_path, cur_cost+next_cost], cur_cost+next_cost)
            # next_cost가 그 노드까지의 총 비용이므로 min pqueue를 통해 '가장 적은 비용'의 UCS 수행
    return cur_path


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    pq = util.PriorityQueue()
    pq.push([problem.getStartState(), [], 0], 0)
    visited = set()
    # 우선순위 큐 사용 A*. (현재 위치, 현재 경로, 전체 비용) 및 f(n) = g(n) + h(n) 기준 min pqueue 선택

    while pq:
        cur_node, cur_path, cur_cost = pq.pop()
        if problem.isGoalState(cur_node): break
        if cur_node in visited: continue
        # visited 집합으로 중복 방문 방지
        visited.add(cur_node)
        successors = problem.getSuccessors(cur_node)
        for successor in successors:
            next_node, next_dir, next_cost = successor
            next_path = cur_path + [next_dir]
            # 현재 노드까지 온 비용 + next_node 비용. next_path는 DFS, BFS와 동일
            pq.push([next_node, next_path, cur_cost + next_cost], cur_cost + next_cost + heuristic(next_node, problem))
            # UCS의 g(n): 이 노드까지의 총 비용 + 휴리스틱 함수의 h(n) = f(n)으로 min pq를 택하는 기준
    return cur_path


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningDepthFirstSearch # ids 사용을 위한 축약어
