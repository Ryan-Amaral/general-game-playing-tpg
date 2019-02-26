import numpy as np
import gym
import gym.spaces
import psutil
import os
import random

# To transform pixel matrix to a single vector.
def getState(inState):
    # each row is all 1 color
    rgbRows = np.reshape(inState,(len(inState[0])*len(inState), 3)).T

    # add each with appropriate shifting
    # get RRRRRRRR GGGGGGGG BBBBBBBB
    return np.add(np.left_shift(rgbRows[0], 16),
        np.add(np.left_shift(rgbRows[1], 8), rgbRows[2]))

"""
Run each agent in this method for parallization.
Args:
    args: (TpgAgent, envName, scoreList, numRepeats, numFrames, vizTrack)
"""
def runAgent(args):
    agent = args[0]
    envName = args[1]
    scoreList = args[2]
    numRepeats = args[3] # number of times to repeat game
    numFrames = args[4]
    visTrack = args[5]

    vTrack = visTrack is not None

    env = gym.make(envName)
    valActs = range(env.action_space.n) # valid actions, some envs are less

    scoreTotal = 0 # score accumulates over all episodes
    for rep in range(numRepeats): # episode loop
        state = env.reset()
        scoreEp = 0
        numRandFrames = random.randint(5,min(20, numFrames))
        for i in range(numFrames): # frame loop
            if i < numRandFrames:
                _, _, isDone, _ = env.step(env.action_space.sample())
                if isDone: # don't count it if lose on random steps
                    rep -= 1
                continue

            act = agent.act(getState(np.array(state, dtype=np.int32)), valActs=valActs,
                    vizd=vTrack)

            # feedback from env
            state, reward, isDone, debug = env.step(act)
            scoreEp += reward # accumulate reward in score
            if isDone:
                break # end early if losing state

        print('Agent #' + str(agent.getAgentNum()) +
              ' | Rep #' + str(rep) + ' | Score: ' + str(scoreEp))
        scoreTotal += scoreEp

    scoreTotal /= numRepeats
    env.close()
    agent.reward(scoreTotal, envName)
    scoreList.append((agent.getUid(), agent.getOutcomes()))
    if vTrack:
        visTrack[agent.team.uid] = agent.screenIndexed

# https://stackoverflow.com/questions/42103367/limit-total-cpu-usage-in-python-multiprocessing/42130713
# probably won't even be used
def limit_cpu():
    p = psutil.Process(os.getpid())
    p.nice(0)
