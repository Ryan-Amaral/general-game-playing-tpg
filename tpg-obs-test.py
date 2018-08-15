from tpg.tpg_trainer import TpgTrainer
from tpg.tpg_agent import TpgAgent

import gym
import gym.spaces

import random

actionSize = 2

trainer = TpgTrainer(actions=range(actionSize), teamPopSizeInit=360)

curScores = [] # hold scores in a generation

def getState():
    state = [0]*actionSize
    idx = random.randint(0, actionSize-1)
    state[idx] = 1
    return state

allStates = []
for i in range(50):
    allStates.append([0,1])
for i in range(50):
    allStates.append([1,0])
    
random.shuffle(allStates)

for gen in range(5000): # generation loop
    curScores = [] # new list per gen
    
    while True: # loop to go through agents
        teamNum = trainer.remainingAgents()
        agent = trainer.getNextAgent()
        if agent is None:
            break # no more agents, so proceed to next gen
        score = 0
        changedAction = False
        for i in range(100): # run episodes that last 200 frames
            state = allStates[i]
            act = agent.act(state) # get action from agent
            if state[act] == 1:
                score += 1
        agent.reward(score) # must reward agent (if didn't already score)
        curScores.append(score) # store score
            
    # at end of generation, make summary of scores
    print(str((gen, min(curScores), max(curScores),
                    sum(curScores)/len(curScores)))) # min, max, avg
    trainer.evolve()
