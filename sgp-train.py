# imports and helper methods
from tpg.tpg_trainer import TpgTrainer
from tpg.tpg_agent import TpgAgent

import gym
import gym.spaces

import multiprocessing as mp
import time
import random
import psutil
import os
import pickle
import operator
import datetime
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-g', '--gen', type='int', dest='curGen', default=0)
(options, args) = parser.parse_args()

"""
inState is (row x col x rgba) list. This converts it to a 1-D list. Because 
that is what TPG uses.
"""
def getState(inState):
    skip = 3
    outState = [0]*(len(inState)*len(inState[0]))
    cnt = 0
    for row in range(0, len(inState), skip):
        for col in range(0, len(inState[row]), skip):
            outState[cnt] = ((inState[row][col][0] >> 1) 
                          + (inState[row][col][1] >> 2) 
                          + (inState[row][col][2] >> 3))
            cnt += 1
    
    return outState[:cnt]

"""
Run each agent in this method for parallization.
Args:
    args: (TpgAgent, envName, scoreList, numEpisodes, numFrames)
"""
def runAgent(args):
    agent = args[0]
    envName = args[1]
    scoreList = args[2]
    numEpisodes = args[3] # number of times to repeat game
    numFrames = args[4] 
    
    # skip if task already done by agent
    if agent.taskDone(envName+'-'+str(numFrames)):
        print('Agent #' + str(agent.getAgentNum()) + ' can skip.')
        scoreList.append((agent.getUid(), agent.getOutcomes()))
        return
    
    env = gym.make(envName)
    valActs = range(env.action_space.n) # valid actions, some envs are less
    
    scoreTotal = 0 # score accumulates over all episodes
    for ep in range(numEpisodes): # episode loop
        state = env.reset()
        scoreEp = 0
        numRandFrames = 0
        if numEpisodes > 1:
            numRandFrames = random.randint(0,30)
        for i in range(numFrames): # frame loop
            if i < numRandFrames:
                _, _, isDone, _ = env.step(env.action_space.sample())
                if isDone: # don't count it if lose on random steps
                    ep -= 1
                continue

            act = agent.act(getState(state), valActs=valActs)

            # feedback from env
            state, reward, isDone, debug = env.step(act)
            scoreEp += reward # accumulate reward in score
            if isDone:
                break # end early if losing state
                
        print('Agent #' + str(agent.getAgentNum()) + 
              ' | Ep #' + str(ep) + ' | Score: ' + str(scoreEp))
        scoreTotal += scoreEp
       
    scoreTotal /= numEpisodes
    env.close()
    agent.reward(scoreTotal, envName+'-'+str(numFrames))
    scoreList.append((agent.getUid(), agent.getOutcomes()))
    
# https://stackoverflow.com/questions/42103367/limit-total-cpu-usage-in-python-multiprocessing/42130713
def limit_cpu():
    p = psutil.Process(os.getpid())
    p.nice(10)

envName = 'Assault-v0'

if options.curGen == 0:
    tmpEnv = gym.make(envName)
    trainer = TpgTrainer(actions=range(tmpEnv.action_space.n), teamPopSizeInit=360)
    tmpEnv.close()
else:
    with open('saved-model-sgp.pkl', 'rb') as f:
        trainer = pickle.load(f)

processes = 2
pool = mp.Pool(processes=processes, initializer=limit_cpu, maxtasksperchild=5)
man = mp.Manager()

allScores = [] # track all scores each generation

tStart = time.time()

curGen = options.curGen # generation of tpg

logFileName = 'sgp-log-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + '.txt'

if curGen < 25:
    frames = 1000
elif curGen < 50:
    frames = 5000
else:
    frames = 18000

while True: # do generations with no end
    curGen += 1
    scoreList = man.list()
    
    agents = trainer.getAllAgents(noRef=True) # swap out agents only at start of generation
       
    if curGen == 1:
        frames = 1000
    elif curGen == 25:
        frames = 5000
    elif curGen == 50:
        frames = 18000
    pool.map(runAgent, 
        [(agent, envName, scoreList, 1, frames)
        for agent in agents])
    
    # apply scores
    trainer.applyScores(scoreList)

    tasks = [envName+'-'+str(frames)]
    scoreStats = trainer.generateScoreStats(tasks=tasks)
    allScores.append((envName, scoreStats['min'], scoreStats['max'], scoreStats['average']))
    trainer.evolve(tasks=tasks, fitShare=False) # go into next gen

    # save model after every gen
    with open('saved-model-sgp.pkl','wb') as f:
        pickle.dump(trainer,f)
    # save best agent after every gen
    with open('best-agent-sgp.pkl','wb') as f:
        pickle.dump(trainer.getBestAgent(tasks=tasks),f)
    
    print('Time Taken (Seconds): ' + str(time.time() - tStart))
    print('On Generation: ' + str(curGen))
        #print('Results: ', str(allScores))

    with open(logFileName, 'a') as f:
        f.write(str(curGen) + ' | ' 
            + str(envName) + ' | ' 
            + str(scoreStats['min']) + ' | ' 
            + str(scoreStats['max']) + ' | '
            + str(scoreStats['average']) + '\n')






