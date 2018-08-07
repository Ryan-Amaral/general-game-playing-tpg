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
    agent.reward(scoreTotal, envName)
    scoreList.append((agent.getUid(), agent.getOutcomes()))
    
# https://stackoverflow.com/questions/42103367/limit-total-cpu-usage-in-python-multiprocessing/42130713
def limit_cpu():
    p = psutil.Process(os.getpid())
    p.nice(10)
    
def getEnvFitness(env, score):
    return (soloTpgScores[env]-score)/(soloTpgScores[env]-soloRandScores[env])
    
# all of the titles we will be general game playing on
# we chose games that we know TPG does OK in alone
allEnvNames = ['Alien-v0','Asteroids-v0','Atlantis-v0','BankHeist-v0',
               'BattleZone-v0','Bowling-v0','Boxing-v0','Centipede-v0',
               'ChopperCommand-v0','DoubleDunk-v0','FishingDerby-v0',
               'Freeway-v0','Frostbite-v0','Gravitar-v0','Hero-v0',
               'IceHockey-v0','Jamesbond-v0','Kangaroo-v0','Krull-v0',
               'KungFuMaster-v0','MsPacman-v0','PrivateEye-v0',
               'RoadRunner-v0','Skiing-v0','Tennis-v0','TimePilot-v0',
               'UpNDown-v0','Venture-v0','WizardOfWor-v0','Zaxxon-v0']

envFitnesses = {}
for envName in allEnvNames:
    envFitnesses[envName] = 0

trainer = TpgTrainer(actions=range(18), teamPopSizeInit=360)

processes = 2
pool = mp.Pool(processes=processes, initializer=limit_cpu)
man = mp.Manager()

envPopSize = 9 # number of envs to up in envNamePop

numEpisodes = 5 # times to evaluate each env
numFrames = 200 # number of frames per episode, to increase as time goes on

allScores = [] # track all scores each generation

tStart = time.time()

envGen = 0 # generation of cycling through env pop

logFileName = 'ggp-log-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + '.txt'
while True: # do generations with no end
    scoreList = man.list() # to hold scores of current gen
    envGen += 1
    if envGen == 5:
        numFrames = 500
    elif envGen == 20:
        numFrames = 1000
    elif envGen == 30:
        numFrames = 2000
    elif envGen == 40:
        numFrames = 5000
    elif envGen == 50:
        numFrames = 18000
    
    # choose the new env name pop
    sortedEnvFits = sorted(envFitnesses.items(), key=operator.itemgetter(1), reverse=True)
    envNamesPop = [envFit[0] for envFit in sortedEnvFits[:envPopSize]]
    
    # run each env multiple times
    for ep in numEpisodes:
        # choose order of envs to run
        envNames = list(envNamesPop)
        random.shuffle(envNames)
        
        # run each env per episode
        for envName in envNames:
            # run the agents in the env
            pool.map(runAgent, 
                [(agent, envName, scoreList, 1, numFrames)
                for agent in trainer.getAllAgents(skipTasks=[], noRef=True)])
        
            trainer.applyScores(scoreList)
            trainer.evolve(tasks=[envName], elitistTasks=allEnvNames)
            
            tpgBest = trainer.scoreStats['max']
            tpgWorst = trainer.scoreStats['min']
            
            # scores of tpg agents normalized between 0 and 1
            tpgScores = [(score-tpgWorst)/(tpgBest-tpgWorst) for score in trainer.scoreStats['scores']]
            
            # calculate fitness of the environments
            envFits = [(2*score/0.75)-(score)**2]
    
    
    
    
    
    
    
    
    
    
    
    
    # reload curGames if needed
    if len(curEnvNames) == 0:
        curCycle += 1
        sortedEnvFits = sorted(envFitnesses.items(), key=operator.itemgetter(1), reverse=True)
        curEnvNames = [envFit[0] for envFit in sortedEnvFits[:numActiveEnvs]]
        random.shuffle(curEnvNames) # not sure if this necessary
        curEnvNamesCp = list(curEnvNames)
        # gradually increase the number of episodes and frames, and decrease active games
        if numEpisodes < 5:
            tmp += 1
            numEpisodes += 1 # up to 5
            numFrames += 200 # up to 1000 go to 18000 actually
            numActiveEnvs -= 4 # down to 9
        if curCycle == cycleSwitch:
            numActiveEnvs = 6 # drop to a reasonable number of titles to play
        
    curEnvName = curEnvNames.pop() # get env to play on this generation
    
    if (curCycle < cycleSwitch or len(curEnvNames) == len(curEnvNamesCp) - 1 or 
            agents is None or len(agents) == 0): # error checking cases
        agents = trainer.getAllAgents(skipTasks=[], noRef=True) # swap out agents only at start of generation
        
    pool.map(runAgent, 
        [(agent, curEnvName, scoreList, numEpisodes, numFrames)
        for agent in agents])
    
    # apply scores
    trainer.applyScores(scoreList)

    if curCycle < cycleSwitch or len(curEnvNames) == 0: # time to evolve
        print(chr(27) + "[2J")
        curGen += 1
        if curCycle < cycleSwitch: # regular evolution, after each individual game
            print('In to new gen!')
            tasks = [curEnvName+'-'+str(numFrames)]
            envsName = curEnvName
        elif len(curEnvNames) == 0: # more advanced, after play all games
            print('In to new cycle!')
            tasks = [envName+'-'+str(numFrames) for envName in curEnvNamesCp]
            envsName = ','.join(curEnvNamesCp)
        
        scoreStats = trainer.generateScoreStats(tasks=tasks)
        allScores.append((envsName, scoreStats['min'], scoreStats['max'], scoreStats['average']))

        # apply fitness of just played env
        if curCycle < cycleSwitch:
            try:
                envFitnesses[curEnvName] = getEnvFitness(curEnvName, scoreStats['average'])
            except:
                envFitnesses[curEnvName] = 1
        elif len(curEnvNames) == 0:
            for env in curEnvNamesCp:
                scoreStats = trainer.generateScoreStats(tasks=[env+'-'+str(numFrames)])
                try:
                    envFitnesses[env] = getEnvFitness(env, scoreStats['average'])
                except:
                    envFitnesses[env] = 1

        trainer.evolve(tasks=tasks) # go into next gen

        # save model after every gen
        with open('saved-model-1.pkl','wb') as f:
            pickle.dump(trainer,f)
    
        print('Time Taken (Seconds): ' + str(time.time() - tStart))
        print('On Generation: ' + str(curGen))
        print('On Cycle: ' + str(curCycle))
        #print('Results: ', str(allScores))

        with open(logFileName, 'a') as f:
            f.write(str(curGen) + ' | ' 
                + str(envsName) + ' | ' 
                + str(scoreStats['min']) + ' | ' 
                + str(scoreStats['max']) + ' | '
                + str(scoreStats['average']) + '\n')
    
    # evaluate env fitnesses, incase haven't visited in a while
    if curCycle % 10 == 0 and len(curEnvNames) == 0: 
        print('In to evaluation of fitnesses of envs!')
        for envName in envNames:
            scoreList = man.list()
            pool.map(runAgent, 
                [(agent, envName, scoreList, numEpisodes, numFrames)
                for agent in agents])
            trainer.applyScores(scoreList)
            scoreStats = trainer.generateScoreStats(tasks=[envName+'-'+str(numFrames)])
            try:
                envFitnesses[envName] = getEnvFitness(env, scoreStats['average'])
            except:
                envFitnesses[envName] = 1
            print('Env ' + envName + ' with fitness ' + str(envFitnesses[envName]))









