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
        
    env.close()
    agent.reward(scoreTotal, envName+'-'+str(numEpisodes)+'-'+str(numFrames))
    scoreList.append((agent.getUid(), agent.getOutcomes()))
    
# https://stackoverflow.com/questions/42103367/limit-total-cpu-usage-in-python-multiprocessing/42130713
def limit_cpu():
    p = psutil.Process(os.getpid())
    p.nice(10)
    
# all of the titles we will be general game playing on
gymEnvNames = ['Alien-v0','Asteroids-v0','Atlantis-v0','BankHeist-v0',
               'BattleZone-v0','Bowling-v0','Boxing-v0','Centipede-v0',
               'ChopperCommand-v0','DoubleDunk-v0','FishingDerby-v0',
               'Freeway-v0','Frostbite-v0','Gravitar-v0','Hero-v0',
               'IceHockey-v0','Jamesbond-v0','Kangaroo-v0','Krull-v0',
               'KungFuMaster-v0','MsPacman-v0','PrivateEye-v0',
               'RoadRunner-v0','Skiing-v0','Tennis-v0','TimePilot-v0',
               'UpNDown-v0','Venture-v0','WizardOfWor-v0','Zaxxon-v0']


trainer = TpgTrainer(actions=range(18), teamPopSizeInit=360)

processes = 2
pool = mp.Pool(processes=processes, initializer=limit_cpu)
man = mp.Manager()

curEnvNames = []
numActiveEnvs = 10

numEpisodes = 0 # repeat evaluations to deal with randomness
numFrames = 250 # number of frames per episode, to increase as time goes on

allScores = [] # track all scores each generation

tStart = time.time()

while True: # do generations with no end
    scoreList = man.list()
    
    # reload curGames if needed
    if len(curEnvNames) == 0:
        curEnvNames = list(gymEnvNames)
        random.shuffle(curEnvNames)
        curEnvNames = curEnvNames[:numActiveEnvs]
        # gradually increase the number of episodes and frames
        if numEpisodes < 5:
            numEpisodes += 1
            numFrames += 150
        
    curEnvName = curEnvNames.pop() # get env to play on this generation
    
    pool.map(runAgent, 
        [(agent, curEnvName, scoreList, numEpisodes, numFrames)
        for agent in trainer.getAllAgents(skipTasks=[])])
    
    # apply scores
    trainer.applyScores(scoreList)
    scoreStats = trainer.generateScoreStats(tasks=[curEnvName+'-'+str(numEpisodes)+'-'+str(numFrames)])
    allScores.append((curEnvName, scoreStats['min'], scoreStats['max'], scoreStats['average']))

    trainer.evolve(tasks=[curEnvName+'-'+str(numEpisodes)+'-'+str(numFrames)]) # go into next gen
    
    # save model after every gen
    with open('saved-model-1.pkl','wb') as f:
        pickle.dump(trainer,f)
        
    print(chr(27) + "[2J")
    print('Time Taken (Seconds): ' + str(time.time() - tStart))
    print('On Generation: ' + str(trainer.curGen))
    print('Results: ', str(allScores))




