import numpy as np
import gym
import gym.spaces
import psutil
import os
import random
import time

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

def champEval(envNames, trainer, lfCName, pool, man, tstart, popName=None, numAgents=5, frames=18000, eps=10):
    print('Evaluating agents on all envs.')
    agents = trainer.getBestAgents(tasks=envNames, popName=popName)[:numAgents]

    agentScores = {} # save scores of agents here
    allVisTrack = {} # save indexed screen space here
    for tId in [agent.team.uid for agent in agents]:
        agentScores[tId] = {}
        allVisTrack[tId] = {}
    for envName in envNames:
        print('On game: ' + envName)
        scoreList = man.list() # reset score list
        visTrack = man.dict()
        
        pool.map(runAgent,
            [(agent, envName, scoreList, eps, frames, visTrack)
            for agent in agents])
        # put scores in dict for env
        for score in scoreList:
            agentScores[score[0]][envName] = score[1][envName]
        for uid in visTrack.keys():
            allVisTrack[uid][envName] = np.array(visTrack[uid])

    for uid in allVisTrack:
        allVisTrack[uid]['visTotal'] = np.zeros(len(allVisTrack[uid][envNames[0]]))
        for envName in envNames:
            for i in range(len(allVisTrack[uid][envName])):
                # visTotal may be slightly off due to environments having different dimensions
                minlen = min(len(allVisTrack[uid]['visTotal']), len(allVisTrack[uid][envName]))
                allVisTrack[uid]['visTotal'] = np.logical_or(allVisTrack[uid]['visTotal'][:minlen],
                        allVisTrack[uid][envName][:minlen])

    with open(lfCName, 'a') as f:
        for uid in agentScores:
            f.write(str(trainer.populations[popName].curGen) + ','
                    + str((time.time()-tstart)/3600) + ','
                    + str(uid))
            for envName in envNames:
                f.write(',' + str(agentScores[uid][envName]))
            for envName in envNames:
                f.write(',' + str(np.count_nonzero(allVisTrack[uid][envName]) / len(allVisTrack[uid][envName])))
            f.write(',' + str(np.count_nonzero(allVisTrack[uid]['visTotal']) / len(allVisTrack[uid]['visTotal'])) + '\n')
