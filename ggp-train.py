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


"""
inState is (row x col x rgba) list. This converts it to a 1-D list. Because 
that is what TPG uses.
"""
def getState(inState):
    outState = [0]*int((len(inState)/5)*(len(inState[0])/5))
    cnt = 0
    for row in range(0, len(inState), 5):
        for col in range(0, len(inState[row]), 5):
            outState[cnt] = ((inState[row][col][0] >> 1) 
                          + (inState[row][col][1] >> 2) 
                          + (inState[row][col][2] >> 3))
            cnt += 1
    
    return outState

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
    if agent.taskDone(envName):
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
    
# all of the titles we will be general game playing on
# we chose games that we know TPG does OK in alone
envNames = ['Alien-v0','Asteroids-v0','Atlantis-v0','BankHeist-v0',
               'BattleZone-v0','Bowling-v0','Boxing-v0','Centipede-v0',
               'ChopperCommand-v0','DoubleDunk-v0','FishingDerby-v0',
               'Freeway-v0','Frostbite-v0','Gravitar-v0','Hero-v0',
               'IceHockey-v0','Jamesbond-v0','Kangaroo-v0','Krull-v0',
               'KungFuMaster-v0','MsPacman-v0','PrivateEye-v0',
               'RoadRunner-v0','Tennis-v0','TimePilot-v0',
               'UpNDown-v0','Venture-v0','WizardOfWor-v0','Zaxxon-v0']

# Kelly, S., & Heywood, M. I. (2018). 
# Emergent Solutions to High-Dimensional Multi-Task Reinforcement Learning. 
# Evolutionary Computation, 26(1), 1–33. 
# https://doi.org/10.1162/evco_a_00232
# scores that tpg achieved running just the game on its own, not ggp
# tennis changed to 1 from 0
soloTpgScores = 
    {'Alien-v0': 3382.7,'Asteroids-v0': 3050.7,'Atlantis-v0': 89653,'BankHeist-v0': 1051,
     'BattleZone-v0': 47233.4,'Bowling-v0': 223.7,'Boxing-v0': 76.5,'Centipede-v0': 34731.7,
     'ChopperCommand-v0': 7070,'DoubleDunk-v0': 2,'FishingDerby-v0': 49,
     'Freeway-v0': 28.9,'Frostbite-v0': 8144.4,'Gravitar-v0': 786.7,'Hero-v0': 16545.4,
     'IceHockey-v0': 10,'Jamesbond-v0': 3120,'Kangaroo-v0': 14780,'Krull-v0': 12850.4,
     'KungFuMaster-v0': 43353.4,'MsPacman-v0': 5156,'PrivateEye-v0': 15028.3,
     'RoadRunner-v0': 17410,'Tennis-v0': 1,'TimePilot-v0': 13540,
     'UpNDown-v0': 34416,'Venture-v0': 576.7,'WizardOfWor-v0': 5196.7,'Zaxxon-v0': 6233.4}

envFitnesses = 
    {'Alien-v0': 0,'Asteroids-v0': 0,'Atlantis-v0': 0,'BankHeist-v0': 0,
    'BattleZone-v0': 0,'Bowling-v0': 0,'Boxing-v0': 0,'Centipede-v0': 0,
    'ChopperCommand-v0': 0,'DoubleDunk-v0': 0,'FishingDerby-v0': 0,
    'Freeway-v0': 0,'Frostbite-v0': 0,'Gravitar-v0': 0,'Hero-v0': 0,
    'IceHockey-v0': 0,'Jamesbond-v0': 0,'Kangaroo-v0': 0,'Krull-v0': 0,
    'KungFuMaster-v0': 0,'MsPacman-v0': 0,'PrivateEye-v0': 0,
    'RoadRunner-v0': 0,'Tennis-v0': 0,'TimePilot-v0': 0,
    'UpNDown-v0': 0,'Venture-v0': 0,'WizardOfWor-v0': 0,'Zaxxon-v0': 0}


trainer = TpgTrainer(actions=range(18), teamPopSizeInit=360)

processes = 2
pool = mp.Pool(processes=processes, initializer=limit_cpu)
man = mp.Manager()

curEnvNames = []
curEnvNamesCp = [] # copy of curEnvNames
numActiveEnvs = 29 #start with all games

numEpisodes = 0 # repeat evaluations to deal with randomness
numFrames = 250 # number of frames per episode, to increase as time goes on

allScores = [] # track all scores each generation

tStart = time.time()

curGen = 0 # generation of tpg
curCycle = 0 # times gone through all current games
cycleSwitch = 100 # switch to play all games in single eval

while True: # do generations with no end
    scoreList = man.list()
    
    # reload curGames if needed
    if len(curEnvNames) == 0:
        curCycle += 1
        sortedEnvFits = sorted(envFitnesses.items(), key=operator.itemgetter(1), reverse=True)
        curEnvNames = [envFit[0] for envFit in sortedEnvFits[:numActiveEnvs]]
        random.shuffle(curEnvNames) # not sure if this necessary
        curEnvNamesCp = list(curEnvNames)
        # gradually increase the number of episodes and frames, and decrease active games
        if numEpisodes < 5:
            numEpisodes += 1 # up to 5
            numFrames += 150 # up to 1000
            numActiveEnvs -= 4 # down to 9
        
    curEnvName = curEnvNames.pop() # get env to play on this generation
    
    if curCycle < cycleSwitch or len(curEnvNames) == len(curEnvNamesCp):
        agents = trainer.getAllAgents(skipTasks=[]) # swap out agents only at start of generation
    pool.map(runAgent, 
        [(agent, curEnvName, scoreList, numEpisodes, numFrames)
        for agent in agents])
    
    # apply scores
    trainer.applyScores(scoreList)

    if curCycle < cycleSwitch or len(curEnvNames) == 0: # time to evolve
        curGen += 1
        if curCycle < cycleSwitch: # regular evolution, after each individual game
            tasks = [curEnvName+'-'+str(numFrames)]
            envsName = curEnvName
        elif len(curEnvNames) == 0: # more advanced, after play all games
            tasks = [envName+'-'+str(numFrames) for envName in curEnvNamesCp]
            envsName = ','.join(curEnvNamesCp)
        
        scoreStats = trainer.generateScoreStats(tasks=tasks)
        allScores.append((envsName, scoreStats['min'], scoreStats['max'], scoreStats['average']))

        # apply fitness of just played env
        if curCycle < cycleSwitch:
            envFitnesses[curEnvName] = (-scoreStats['average']/soloTpgScores[curEnvName]) + 1
        elif len(curEnvNames) == 0:
            for env in curEnvNamesCp:
                scoreStats = trainer.generateScoreStats(tasks=[env+'-'+str(numFrames)])
                envFitnesses[env] = (-scoreStats['average']/soloTpgScores[env]) + 1

        trainer.evolve(tasks=tasks) # go into next gen

        # save model after every gen
        with open('saved-model-1.pkl','wb') as f:
            pickle.dump(trainer,f)
    
        print(chr(27) + "[2J")
        print('Time Taken (Seconds): ' + str(time.time() - tStart))
        print('On Generation: ' + str(curGen))
        print('On Cycle: ' + str(curCycle))
        print('Results: ', str(allScores))
            

