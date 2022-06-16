# 2016130927.py
# ---------------
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
import game
from game import Agent
from searchProblems import PositionSearchProblem

import util
import time
import search

"""
IMPORTANT
`agent` defines which agent you will use. By default, it is set to ClosestDotAgent,
but when you're ready to test your own agent, replace it with MyAgent
"""
def createAgents(num_pacmen, agent='MyAgent'):
    # agent 지정을 MyAgent로 변경 완료
    return [eval(agent)(index=i) for i in range(num_pacmen)]

eaten = set()
# 각 팩맨이 경로 상 먹은 food의 좌표를 보관하는 전역 변수 집합 eaten

class MyAgent(Agent):
    """
    Implementation of your agent.
    """
    def findPathToMoreDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition(self.index)
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState, self.index)

        x, y = problem.getStartState()
        cur_food = 0
        if food[x][y] and problem.getStartState() not in eaten:
            eaten.add(problem.getStartState())
            cur_food -= 10
        #   min-pq에 따라 cur_food는 음수로 기준

        pacmanCurrent = [problem.getStartState(), [], 0, cur_food + 0]
        # 경로 비용이 아닌 경로 상의 food의 개수가 우선순위 큐의 기준
        visited = set()
        visited.add(problem.getStartState())
        #   eaten은 전역, visited는 로컬(각 팩맨의 입장에서 고려해야 함)
        pq = util.PriorityQueue()
        pq.push(pacmanCurrent, cur_food)
        # 초기 상태 pq에 push

        while pq:
            pacmanCurrent = pq.pop()
            cur_state, cur_path, cur_cost, cur_food = pacmanCurrent

            if problem.isGoalState(cur_state): return cur_path

            pacmanSuccessors = problem.getSuccessors(cur_state)
            for next_state, next_dir, next_cost in pacmanSuccessors:
                x, y = next_state
                next_food = cur_food

                if next_state not in visited:
                    visited.add(next_state)
                    next_path = cur_path + [next_dir]
                    next_cost += cur_cost
                    if food[x][y] and next_state not in eaten:
                        # 다음 위치에 food가 존재하고 아직 다른 팩맨이 먹지 않았을 때
                        eaten.add(next_state)
                        next_food -= 10
                        # 전역 변수 eaten에 추가 및 우선도를 높이기 위해 next_food 값 조정
                        manhattan_distance = abs(cur_state[0] - x) + abs(cur_state[1] - y)
                        next_food -= 1/manhattan_distance
                        # 주어진 food 중에서 이동하기 가장 가까우리라 생각되는 곳으로 이동하기: 맨해튼 거리를 통해 relaxed problem 풀기
                        # pq의 앞에서 pop되는 좌표는 (food가 존재한다면) 맨해튼 거리를 통해 우선순위 보정
                    pq.push([next_state, next_path, next_cost, next_food], next_cost + next_food)
                    # 기본적으로 경로 이동 비용 next_cost를 통해 이동 / next_food를 통해 경로 상 존재하는 food의 개수에 따라 우선순위 조정

        return cur_path

    def getAction(self, state):
        """
        Returns the next action the agent will take
        """
        return self.findPathToMoreDot(state)[0]

    def initialize(self):
        """
        Intialize anything you want to here. This function is called
        when the agent is first created. If you don't need to use it, then
        leave it blank
        """

        "*** YOUR CODE HERE"

        # raise NotImplementedError()

"""
Put any other SearchProblems or search methods below. You may also import classes/methods in
search.py and searchProblems.py. (ClosestDotAgent as an example below)
"""

class ClosestDotAgent(Agent):

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition(self.index)
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState, self.index)


        "*** YOUR CODE HERE ***"

        pacmanCurrent = [problem.getStartState(), [], 0]
        visitedPosition = set()
        # visitedPosition.add(problem.getStartState())
        fringe = util.PriorityQueue()
        fringe.push(pacmanCurrent, pacmanCurrent[2])
        while not fringe.isEmpty():
            pacmanCurrent = fringe.pop()
            if pacmanCurrent[0] in visitedPosition:
                continue
            else:
                visitedPosition.add(pacmanCurrent[0])
            if problem.isGoalState(pacmanCurrent[0]):
                return pacmanCurrent[1]
            else:
                pacmanSuccessors = problem.getSuccessors(pacmanCurrent[0])
            Successor = []
            for item in pacmanSuccessors:  # item: [(x,y), 'direction', cost]
                if item[0] not in visitedPosition:
                    pacmanRoute = pacmanCurrent[1].copy()
                    pacmanRoute.append(item[1])
                    sumCost = pacmanCurrent[2]
                    Successor.append([item[0], pacmanRoute, sumCost + item[2]])
            for item in Successor:
                fringe.push(item, item[2])
        return pacmanCurrent[1]

    def getAction(self, state):
        return self.findPathToClosestDot(state)[0]

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState, agentIndex):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state
        if self.food[x][y] == True:
            return True
        return False

