from your_baseline2 import *

def createTeam(firstIndex, secondIndex, isRed,
               first='AQLearningAgent', second='TimidAgent'):
  # Approximate Q-Learning Agent가 주로 Offensive
  return [eval(first)(firstIndex), eval(second)(secondIndex)]
