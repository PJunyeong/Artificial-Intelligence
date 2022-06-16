from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions, Actions
import game
from util import nearestPoint
from collections import deque

def createTeam(firstIndex, secondIndex, isRed,
               first='BaseAgent', second='TimidAgent'):
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

class BaseAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.caught = False
    self.capsules = len(self.getCapsules(gameState))
    self.scared = 0
    self.min_ghost_distance = 5
    # (1). 적의 고스트에게 쫓기고 있는지 (2). 존재하는 캡슐의 개수 (3). 캡슐 섭취 뒤 시간 카운트

  def is_caught(self, gameState):
    if not gameState.getAgentState(self.index).isPacman or self.scared > 0: return False
    cur_pos = gameState.getAgentState(self.index).getPosition()
    enemies = [gameState.getAgentState(opponent) for opponent in self.getOpponents(gameState)]
    ghosts = [enemy for enemy in enemies if not enemy.isPacman and enemy.getPosition() != None]

    for ghost in ghosts:
      distance = self.getMazeDistance(cur_pos, ghost.getPosition())
      if distance < self.min_ghost_distance: return True
        # 상대방 고스트에게 쫓기는 거리가 self.min_ghost_distance 이하이면 쫓긴다고 판단
    return False

  def chooseAction(self, gameState):
    # baseline과 마찬가지로 가장 큰 Q 값 액션 리턴
    actions = gameState.getLegalActions(self.index)

    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction


    if len(self.getCapsules(gameState)) < self.capsules:
      # 현재 에이전트가 캡슐을 먹은 상태 -> 상대방은 겁 먹은 상태
      self.capsules -= 1
      self.scared = 20

    if self.scared > 0: self.scared -= 1
    # 캡슐은 이전에 먹었다면  -> 시간 카운트
    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def is_current_Pacman(self, gameState):
    # 현재 에이전트가 팩맨이라면 True 리턴
    if gameState.getAgentState(self.index).isPacman: return True
    else: return False

  def is_eating_Pacman(self, gameState, action):
    # successor가 음식을 먹는다면 True 리턴
    successor = self.getSuccessor(gameState, action)
    if not self.is_current_Pacman(successor): return False

    cur_food_num = len(self.getFood(gameState).asList())
    next_food_num = len(self.getFood(successor).asList())

    if cur_food_num > next_food_num: return True
    else: return False
    # successor의 음식 개수가 현재 상태의 개수보다 더 '적다면' 음식을 먹고 있다는 뜻

  def is_eaten_Pacman(self, gameState, action):
    # 다음 번 팩맨이 '먹힐 때' True 리턴
    if not self.is_current_Pacman(gameState): return False

    cur_pos = gameState.getAgentPosition(self.index)
    successor = self.getSuccessor(gameState, action)
    next_pos = successor.getAgentState(self.index).getPosition()

    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghosts = [ghost for ghost in enemies if not ghost.isPacman and ghost.getPosition() != None]

    if ghosts:
      distance_to_ghosts = [self.getMazeDistance(cur_pos, ghost.getPosition()) for ghost in ghosts]
      if min(distance_to_ghosts) == 1 and next_pos == self.start: return True

    return False

  def get_nearest_food(self, gameState, action):
    # 가장 가까운 음식까지 걸리는 비용 리턴
    foods = self.getFood(gameState)
    cur_pos = gameState.getAgentState(self.index).getPosition()
    walls = gameState.getWalls()

    successor = self.getSuccessor(gameState, action)
    next_pos = successor.getAgentState(self.index).getPosition()
    x, y = next_pos
    x, y = int(x), int(y)

    queue = deque()
    visited = set()
    queue.append([x, y, 0])
    visited.add((x, y))

    while queue:
      x, y, distance = queue.popleft()
      if foods[x][y]:
        value = float(distance) / (walls.width * walls.height)
        return (True, value)
      next_positions = Actions.getLegalNeighbors((x, y), walls)
      for next_x, next_y in next_positions:
        if not (next_x, next_y) in visited:
          visited.add((next_x, next_y))
          queue.append([next_x, next_y, distance + 1])

    return (False, -1)

  def get_ghosts_behind(self, gameState, action):
    # 바로 뒤에 팩맨이 고스트에게 잡히면 True, 그렇지 않는다면 False를 리턴.
    foods = self.getFood(gameState)
    enemies = [gameState.getAgentState(opponent) for opponent in self.getOpponents(gameState)]
    ghosts = [enemy for enemy in enemies if not enemy.isPacman and enemy.getPosition() != None]
    cur_pos = gameState.getAgentPosition(self.index)
    x, y = cur_pos

    successor = self.getSuccessor(gameState, action)
    next_pos = successor.getAgentPosition(self.index)
    next_x, next_y = next_pos
    next_x, next_y = int(next_x), int(next_y)

    walls = gameState.getWalls()

    num_ghosts = 0
    for ghost in ghosts:
      if (next_x, next_y) in Actions.getLegalNeighbors(ghost.getPosition(), walls): num_ghosts += 1

    if num_ghosts > 0: return (True, num_ghosts)
    elif num_ghosts == 0 and foods[next_x][next_y]: return (False, 1.0)
    else: return (False, 0.0)

  def get_ghost_info(self, gameState, action):
    # 근처 고스트까지의 최단 맨해튼 거리를 리턴
    successor = self.getSuccessor(gameState, action)
    next_pos = successor.getAgentPosition(self.index)
    enemies = [successor.getAgentState(opponent) for opponent in self.getOpponents(successor)]
    ghosts = [enemy for enemy in enemies if not enemy.isPacman and enemy.getPosition() != None]
    num_ghost = len(ghosts)
    distance_to_ghost = 0
    if ghosts: distance_to_ghost = min([self.getMazeDistance(next_pos, ghost.getPosition()) for ghost in ghosts])

    return num_ghost, distance_to_ghost

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    next_pos = successor.getAgentState(self.index).getPosition()

    if self.caught:
      distance_to_home = self.getMazeDistance(next_pos, self.start)
      # 상대방 고스트에게 쫓기고 있다면 집으로 가는 비용을 늘린다.
      features['distanceToHome'] = distance_to_home

    if action == Directions.STOP:
      stop = 1
      features['stop'] = stop

    if action == Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]:
      reverse = 1
      features['reverse'] = reverse
      # baseline과 상동

    flag, food_distance = self.get_nearest_food(gameState, action)
    if flag: features["distanceToFood"] = food_distance

    if self.is_current_Pacman(gameState):
      flag, result = self.get_ghosts_behind(gameState, action)
      if flag: features['ghostBehind'] = result
      else:
        features['isFoodEaten'] = result
      if self.scared != 0:
        num_ghost, distance_to_ghost = self.get_ghost_info(gameState, action)
        features['numGhost'] = num_ghost
        features['distanceToGhost'] = distance_to_ghost
      # 고스트까지의 최단 맨해튼 거리 리턴
      # 현재 에이전트가 팩맨이고 쫓기는/먹이를 먹는 상황 리턴


    else:
      enemies = [successor.getAgentState(opponent) for opponent in self.getOpponents(successor)]
      invaders = [invader for invader in enemies if invader.isPacman and invader.getPosition() != None]
      features['numInvaders'] = len(invaders)
      # 현재 에이전트가 고스트라면 상대 팩맨과의 최단 거리를 리턴
      if invaders:
        distance_to_invader = sys.maxsize
        for invader in invaders: distance_to_invader = min(distance_to_invader,
                                                           self.getMazeDistance(next_pos, invader.getPosition()))
        features['distanceToInvader'] = distance_to_invader

    self.caught = self.is_caught(gameState)
    # 현재 쫓기고 있는지 업데이트

    return features

  def getWeights(self, gameState, action):
    return {'successorScore': -50, 'distanceToHome': -500, 'stop': -1000, 'reverse': -2, 'isFoodEaten': 30,
            'distanceToFood': -100, 'numGhost': -10, 'distanceToGhost': 3, 'ghostBehind': -30, 'numInvaders': 0, 'distanceToInvader': 0}
    # TODO: 다시 한번 가중치 확인하기!
    # numInvader, distanceToInvader == 0 -> 에이전트 = 팩맨일 때 더 많은 음식을 먹는 데 집중한다

class DefensiveAgent(BaseAgent):

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    next_pos = successor.getAgentState(self.index).getPosition()

    if self.caught:
      distance_to_home = self.getMazeDistance(next_pos, self.start)
      # 상대방 고스트에게 쫓기고 있다면 집으로 가는 비용을 늘린다.
      features['distanceToHome'] = distance_to_home

    if action == Directions.STOP:
      stop = 1
      features['stop'] = stop

    if action == Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]:
      reverse = 1
      features['reverse'] = reverse
      # baseline과 상동

    flag, food_distance = self.get_nearest_food(gameState, action)
    if flag: features["distanceToFood"] = food_distance

    if self.is_current_Pacman(gameState):
      flag, result = self.get_ghosts_behind(gameState, action)
      if flag: features['ghostBehind'] = result
      else:
        features['isFoodEaten'] = result
      if self.scared != 0:
        num_ghost, distance_to_ghost = self.get_ghost_info(gameState, action)
        features['numGhost'] = num_ghost
        features['distanceToGhost'] = distance_to_ghost
      # 고스트까지의 최단 맨해튼 거리 리턴
      # 현재 에이전트가 팩맨이고 쫓기는/먹이를 먹는 상황 리턴

    enemies = [successor.getAgentState(opponent) for opponent in self.getOpponents(successor)]
    invaders = [invader for invader in enemies if invader.isPacman and invader.getPosition() != None]
    features['numInvaders'] = len(invaders)
    # 현재 에이전트가 고스트라면 상대 팩맨과의 최단 거리를 리턴
    if invaders:
      distance_to_invader = sys.maxsize
      for invader in invaders: distance_to_invader = min(distance_to_invader,
                                                         self.getMazeDistance(next_pos, invader.getPosition()))
      features['distanceToInvader'] = distance_to_invader

    self.caught = self.is_caught(gameState)
    # 현재 쫓기고 있는지 업데이트

    return features


  def getWeights(self, gameState, action):
    return {'successorScore': -100, 'distanceToHome': -500, 'stop': -1000, 'reverse': -2, 'isFoodEaten': 10,
            'distanceToFood': -5, 'numGhost': 0, 'distanceToGhost': 0, 'ghostBehind': -10, 'numInvaders': -20, 'distanceToInvader': -20}
    # 위 BaseAgent보다 에이전트 = 고스트일 때 상대방 팩맨을 잡는 데 집중한다.

class TimidAgent(DefensiveAgent):

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.caught = False
    self.capsules = len(self.getCapsules(gameState))
    self.scared = 0
    self.min_ghost_distance = 10
    # (1). 적의 고스트에게 쫓기고 있는지 (2). 존재하는 캡슐의 개수 (3). 캡슐 섭취 뒤 시간 카운트


  def getWeights(self, gameState, action):
    return {'successorScore': -100, 'distanceToHome': -1000, 'stop': -1000, 'reverse': -2, 'isFoodEaten': 10,
            'distanceToFood': -10, 'numGhost': -10, 'distanceToGhost': -5, 'ghostBehind': -10, 'numInvaders': -20, 'distanceToInvader': -20}
    # 위 BaseAgent보다 에이전트 = 팩맨일 때 다시 자신의 영토로 빠르게 돌아올 확률이 커진다.