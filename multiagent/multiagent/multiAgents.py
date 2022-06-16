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
import sys

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
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # 팩맨의 다음 위치, 남은 음식, 고스트의 위치, 고스트가 멈추는 시간 등을 통해 "더 좋은" 정보인지 평가하는 함수
        # 팩맨이 가장 가까운 음식과 거리가 가깝고 고스트와 멀리 있다면 더 큰 휴리스틱 값을 리턴하자

        heuristic, INF = 0, sys.maxsize
        food_distance = INF
        for food_pos in newFood.asList():
            food_distance = min(food_distance, manhattanDistance(newPos, food_pos))
            # 음식 거리: 다음 위치와 맨해튼 거리가 가장 가까운 음식까지의 맨해튼 거리

        if food_distance == 0: food_heuristic = 1
        # 맨해튼 거리가 0이면 곧 다음 위치가 음식. 휴리스틱 값을 크게 준다.
        else: food_heuristic = 1 / food_distance
        # 거리가 멀면 가중치가 매우 줄어드는 역수를 취한 상태
        # food_distance가 초깃값으로 바뀌지 않았을 때(즉 newFood가 더 이상 없을 때) 게임 승리

        ghost_distance = INF
        for ghostState in newGhostStates:
            local_ghost_distance = manhattanDistance(newPos, ghostState.configuration.pos)
            # 고스트 거리: 다음 위치와 맨해튼 거리가 가장 가까운 고스트까지의 맨해튼 거리
            if local_ghost_distance < 3 and ghostState.scaredTimer > 1 and food_distance < 2: continue
            # 고스트 거리가 매우 가깝지만(즉 맨해튼 거리 값이 작을 때) 현재 가장 가까운 음식까지의 거리가 짧고 음식을 통해 이 고스트를 멈출 수 있다면 도전 가능
            # 고스트 타이머 조건을 제외하면 팩맨이 "너무 과용을 부리기" 때문에 중도 실패.
            ghost_distance = min(ghost_distance, local_ghost_distance)

        if ghost_distance < 3: ghost_heuristic = -11
        # 가까운 음식이 없고 음식을 먹어도 오랫 동안 고스트를 멈출 수 없는 시점에 고스트가 매우 가까이 있는 상태.
        # 휴리스틱 값을 매우 큰 음수로 준다. 음식을 먹어 유틸리티를 올리는 것보다 고스트를 피해 사망을 피하는 게 더 우선순위가 높기 때문.
        else: ghost_heuristic = 1 / ghost_distance
        # 거리가 멀면 가중치가 매우 줄어드는 역수를 취한 상태

        heuristic += food_heuristic + ghost_heuristic + successorGameState.getScore()
        # 음식과 고스트 간의 거리 및 조건을 고려해 다음 게임 상황에서 팩맨이 얻을 점수를 휴리스틱으로 사용한다.
        # (힌트: 휴리스틱에 음식/고스트 거리를 원값이 아니라 역수값을 취해 사용하라고 권함)
        return heuristic

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
        max_val, max_action = self.maximizer(gameState, self.depth, 0)
        # minimax 시작. 팩맨(인덱스=0)은 현재 위치에서 최댓값을 리턴하는 maximizer 사용
        return max_action

    def minimizer(self, gameState, depth, index):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), ''
        # 터미널 상태(리프-루트 노드까지 연산 완료/승패 결정)에서 상태 값 리턴
        min_val, min_action = sys.maxsize, ''

        for action in gameState.getLegalActions(index):
            successor_state = gameState.generateSuccessor(index, action)
            # 현재 agent(인덱스)가 해당 action 했을 때 다음 successor 상태

            if index == gameState.getNumAgents() - 1:
                # 현재 agent가 마지막 고스트일 때, 즉 모든 고스트를 다 본 상태
                cur_val, _ = self.maximizer(successor_state, depth - 1, 0)
                # 레벨을 한층 높인 depth-1에서 다시 시작. 인덱스=0이 팩맨 에이전트이기 때문에 maximizer 호출
            else:
                # 현재 agent는 고스트이기 때문에 최솟값을 리턴하는 minimizer 호출
                cur_val, _ = self.minimizer(successor_state, depth, index + 1)
            if min_val > cur_val:
                min_val = cur_val
                min_action = action
                # 터미널 노드에서부터 현 깊이까지 최솟값 및 최솟값으로 이어지는 action 기록

        return min_val, min_action

    def maximizer(self, gameState, depth, index):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), ''
        # 터미널 상태(리프-루트 노드까지 연산 완료/승패 결정)에서 상태 값 리턴
        max_val, max_action = -sys.maxsize, ''

        for action in gameState.getLegalActions(index):
            successor_state = gameState.generateSuccessor(index, action)
            # 현재 agent(인덱스)가 해당 action 했을 때 다음 successor 상태
            cur_val, _ = self.minimizer(successor_state, depth, index+1)
            # 인덱스=0:팩맨 / 인덱스 1~n:고스트므로 minimizer 호출.
            if max_val < cur_val:
                max_val = cur_val
                max_action = action
                # 터미널 노드에서 현 깊이까지 최댓값 및 최댓값으로 이어지는 action 기록

        return max_val, max_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        max_val, max_action = self.maximizer(gameState, self.depth, -sys.maxsize, sys.maxsize, 0)
        # alpha(maximizer가 리프~현재 깊이까지 가진 최댓값), beta(minimizer가 리프~현재 깊이까지 가진 최솟값) 값 통해 pruning.
        # 최댓값 갱신 위해 alpha == -INF, beta == +INF로 초기화.
        # minimax 시작. 팩맨(인덱스=0)은 현재 위치에서 최댓값을 리턴하는 maximizer 사용 (minimax 알고리즘과 flow 동일)
        return max_action

    # beta 갱신 및 alpha 비교를 통한 pruning 제외하고는 minimax의 minimizer와 상동
    def minimizer(self, game_state, depth, alpha, beta, index):
        if depth == 0 or game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state), ''
        min_val = sys.maxsize

        for action in game_state.getLegalActions(index):
            successor_game_state = game_state.generateSuccessor(index, action)

            if index == game_state.getNumAgents() - 1:
                cur_val, _ = self.maximizer(successor_game_state, depth-1, alpha, beta, 0)
            else:
                cur_val, _ = self.minimizer(successor_game_state, depth, alpha, beta, index + 1)

            if min_val > cur_val:
                min_val = cur_val
                min_action = action
            beta = min(beta, min_val)
            # beta는 리프 노드에서 현재 레벨까지의 minimizer가 택할 수 있는 최솟값
            if min_val < alpha: return min_val, min_action
            # pruning. mazimizer의 alpha보다 현재 최솟값이 작다면 더 이상 연산할 필요가 없다.
            # alpha보다 작은 값이 있다 해도 mazimizer가 받지 않기 때문.

        return min_val, min_action
        # pruning이 불가능했기 때문에 모든 하위 트리를 살펴본 경우. 일반적인 minimizer가 리턴하는 시점.

    # alpha 갱신 및 beta 비교를 통한 pruning 제외하고는 minimax의 maximizer와 상동
    def maximizer(self, game_state, depth, alpha, beta, index):
        if depth == 0 or game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state), ''
        max_val = -sys.maxsize

        for action in game_state.getLegalActions(index):
            successor_game_state = game_state.generateSuccessor(index, action)
            cur_val, _ = self.minimizer(successor_game_state, depth, alpha, beta, index + 1)

            if max_val < cur_val:
                max_val = cur_val
                max_action = action
            alpha = max(alpha, max_val)
            # 알파는 맥시마이저가 가지고 있는 최댓값으로 유지
            if beta < max_val: return max_val, max_action
            # pruning. minimizer의 beta보다 현재 최댓값이 크다면 더 이상 연산할 필요가 없다.
            # beta보다 큰 값이 있다 해도 minimizer가 받지 않기 때문.

        return max_val, max_action
        # pruning이 불가능했기 때문에 모든 하위 트리를 살펴본 경우. 일반적인 maximizer가 리턴하는 시점.


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
