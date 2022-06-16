from your_baseline1 import *
from baseline import *

EPSILON = 0.4
LEARNING_RATE = 0.2
DISCOUNT = 0.9
# 학습률, 감마(디스카운트) 값 전역 선언

def createTeam(firstIndex, secondIndex, isRed,
               first='AQLearningAgent', second='DefensiveReflexAgent'):
  # Approximate Q-Learning Agent가 주로 Offensive, 기존 ReflexAgent(Baseline) 활용
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

class QLearningAgent(BaseAgent):
  # Q value 학습을 위한 QLearning 에이전트

  def registerInitialState(self, gameState):
    self.epsilon = EPSILON
    self.learning_rate = LEARNING_RATE
    self.discount = DISCOUNT
    self.start = gameState.getAgentPosition(self.index)
    self.caught = False
    self.weights = {'distanceToFood': -5,
                    'bias': -10,
                    'numGhost': -10,
                    'ghostBehind': -10,
                    'distanceToGhost': 3,
                    'distanceToHome': -10,
                    'isFoodEaten': 10}
    # weights를 계속해서 업데이트해서 해야 하므로 에이전트의 self에 변화를 기록
    self.q_value = util.Counter()
    self.last_states = []
    self.last_actions = []
    # Q 값 업데이트를 위한 지난 마지막 상태/행동 정보
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    actions = gameState.getLegalActions(self.index)
    if not actions: return None

    if self.last_states:
      last_state = self.last_states[-1]
      last_action = self.last_actions[-1]
      next_state = self.getSuccessor(last_state, last_action)
      reward = self.get_reward(last_state, last_action)
      self.update(last_state, last_action, next_state, reward)
      # 업데이트할 최신 정보(상태/행동) -> 리워드 계산 + Q값 업데이트

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

    if util.flipCoin(self.epsilon): action = random.choice(actions)
    else: action = self.get_policy(gameState)
    # 확률적으로 정책/랜덤 행동

    self.last_states.append(gameState)
    self.last_actions.append(action)

    return action

  def getFeatures(self, gameState, action):
    features = util.Counter()
    features["bias"] = 1.0

    flag, result = self.get_ghosts_behind(gameState, action)
    if flag: features['ghostBehind'] = result
    else: features['isFoodEaten'] = result
    # 현재 에이전트 = 팩맨일 때 음식을 먹어도 되는지 리턴

    # if self.is_current_Pacman(gameState):
    #   num_ghost, distance_to_ghost = self.get_ghost_info(gameState, action)
    #   features['numGhost'] = num_ghost
    #   features['distanceToGhost'] = distance_to_ghost
      # 현재 에이전트 = 팩맨일 때 고스트까지의 최단 거리 리턴

    flag, food_distance = self.get_nearest_food(gameState, action)
    if flag: features["distanceToFood"] = food_distance
    # 음식까지 걸리는 최소 비용 리턴

    for key in features.keys(): features[key] = features[key] / 10
    # 특성값 정규화

    return features

  def get_QValue(self, gameState, action):
    return self.q_value[(gameState, action)]

  def update(self, gameState, action, successor, reward):
    old_value = self.get_QValue(gameState, action)
    future_value = self.get_value(successor)
    difference = (reward + self.discount * future_value) - old_value
    self.q_value[(gameState, action)] = old_value + self.learning_rate * difference
    # MDP Q value 업데이트: 이전 값과 이후 값 간의 차(difference)에 학습률을 곱한 값을 이전 값에 더한 게 정확한 Q 값

  def get_reward(self, gameState, action):
    reward = 0
    nextState = self.getSuccessor(gameState, action)

    if self.getScore(nextState) > self.getScore(gameState):
      diff = self.getScore(nextState) - self.getScore(gameState)
      reward += diff * 20

    flag, distance_to_food = self.get_nearest_food(gameState, action)
    if flag: reward = distance_to_food

    if self.is_eaten_Pacman(gameState, action): reward -= 100
    # 다음 행동으로 이어지는 결괏값에 따라 리워드 값을 조정 (먹힌다면 패널티 부과, 음식을 먹는다면 상점 추가 등)
    return reward

  def get_value(self, gameState):
    actions = gameState.getLegalActions(self.index)
    if not actions: return 0.0
    policy = self.get_policy(gameState)
    return self.get_QValue(gameState, policy)
  # 현재 행동 가능하다면 현 상태에 적절한 정책(policy)를 가져올 수 있음. 이 정책으로 기록된 Q 값을 리턴.

  def get_policy(self, gameState):
    actions = gameState.getLegalActions(self.index)
    if not actions: return None
    policies = [(self.get_QValue(gameState, action), action) for action in actions]
    policies.sort(key=lambda x:-x[0])
    max_q, _ = policies[0]
    policies = [action for q_val, action in policies if q_val == max_q]
    return random.choice(policies)
  # 현재 행동 가능하다면 이 행동으로 이어지는 정책을 모두 리턴 가능. 이 정책으로 만들어지는 Q 값이 최댓값인 정책 중 하나를 랜덤으로 리턴

class AQLearningAgent(BaseAgent):
  def get_QValue(self, gameState, action):
    features = self.getFeatures(gameState, action)
    return features * self.weights
  # 기존 BaseAgent의 self.evalueate와 상동

  def update(self, gameState, action, successor, reward):
    features = self.getFeatures(gameState, action)
    old_value = self.get_QValue(gameState, action)
    future_value = self.getValue(successor)
    difference = (reward + self.discount * future_value) - old_value

    for feature in features: self.weights[feature] += self.learning_rate * difference * features[feature]
    # Q 값을 정확하게 계산하지 않고 Approximate하게 연산. difference와 가중치 값을 곱한 값을 누적한다.
