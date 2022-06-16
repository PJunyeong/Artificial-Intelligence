from your_baseline3 import *

def createTeam(firstIndex, secondIndex, isRed,
               first='AQLearningAgent', second='DefensiveAgent'):
  # Approximate Q-Learing Agent가 Offensive, TimidAgent (BaseAgent 개조)가 Defensive
  return [eval(first)(firstIndex), eval(second)(secondIndex)]
