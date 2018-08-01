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


"""
inState is (row x col x rgba) list. This converts it to a 1-D list. Because 
that is what TPG uses.
"""
def getState(inState):
    outState = []
    for row in inState:
        for cell in row:
            outState.append(cell[0]/8 + cell[1]*4 + cell[2]*128)
    
    return outState

"""
Run each agent in this method for parallization.
Args:
    args: (TpgAgent, envName, scoreList, numEpisodes)
"""
def runAgent(args):
    agent = args[0]
    envName = args[1]
    scoreList = args[2]
    numEpisodes = args[3] # number of times to repeat game
    
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
        for i in range(1000): # frame loop
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
    agent.reward(scoreTotal, envName)
    scoreList.append((agent.getUid(), agent.getOutcomes()))
    
# https://stackoverflow.com/questions/42103367/limit-total-cpu-usage-in-python-multiprocessing/42130713
def limit_cpu():
    p = psutil.Process(os.getpid())
    p.nice(19)
    
# all of the titles we will be general game playing on
gymEnvNames = ['Alien-v0','Asteroids-v0','Atlantis-v0','BankHeist-v0',
               'BattleZone-v0','Bowling-v0','Boxing-v0','Centipede-v0',
               'ChopperCommand-v0','DoubleDunk-v0','FishingDerby-v0',
               'Freeway-v0','Frostbite-v0','Gravitar-v0','Hero-v0',
               'IceHockey-v0','Jamesbond-v0','Kangaroo-v0','Krull-v0',
               'KungFuMaster-v0','MsPacman-v0','PrivateEye-v0',
               'RoadRunner-v0','Skiing-v0','Tennis-v0','TimePilot-v0',
               'UpNDown-v0','Venture-v0','WizardOfWor-v0','Zaxxon-v0']


trainer = TpgTrainer(actions=range(18))

processes = 5
pool = mp.Pool(processes=processes, initializer=limit_cpu)
man = mp.Manager()

curEnvs = []
numActiveEnvs = 10

numEpisodes = 5

while True: # do generations with no end
    scoreList = man.list()
    
    # reload curGames if needed
    if len(curEnvs) == 0:
        curEnvs = list(gymEnvNames)
        random.shuffle(curEnvs)
        curEnvs = curEnvs[:numActiveEnvs]
        
    curEnv = curEnvs.pop() # get env to play on this generation
    
    pool.map(runAgent, 
        [(agent, curEnv, scoreList, numEpisodes)
        for agent in trainer.getAllAgents(skipTasks=[])])
    
    # apply scores
    trainer.applyScores(scoreList)
    trainer.evolve() # go into next gen
    
    # save model after every gen
    with open('saved-model-1.pkl','wb') as f:
        pickle.dump(trainer,f)
        
    clear_output(wait=True)
    print('Time Taken (Seconds): ' + str(time.time() - tStart))
    print('On Generation: ' + trainer.curGen)
    print('Results so far: ' + 
          str(trainer.generateScoreStats(tasks=['curEnv'])))