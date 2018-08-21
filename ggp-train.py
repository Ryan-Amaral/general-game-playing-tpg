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
parser.add_option('-e', '--envgen', type='int', dest='envGen', default=0)
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
    args: (TpgAgent, envName, scoreList, numRepeats, numFrames)
"""
def runAgent(args):
    agent = args[0]
    envName = args[1]
    scoreList = args[2]
    numRepeats = args[3] # number of times to repeat game
    numFrames = args[4]
    
    # skip if task already done by agent
    if agent.taskDone(envName):
        print('Agent #' + str(agent.getAgentNum()) + ' can skip.')
        scoreList.append((agent.getUid(), agent.getOutcomes()))
        return
    
    env = gym.make(envName)
    valActs = range(env.action_space.n) # valid actions, some envs are less
    
    scoreTotal = 0 # score accumulates over all episodes
    for rep in range(numRepeats): # episode loop
        state = env.reset()
        scoreEp = 0
        numRandFrames = 0
        if numRepeats > 1:
            numRandFrames = random.randint(0,30)
        for i in range(numFrames): # frame loop
            if i < numRandFrames:
                _, _, isDone, _ = env.step(env.action_space.sample())
                if isDone: # don't count it if lose on random steps
                    rep -= 1
                continue

            act = agent.act(getState(state), valActs=valActs)

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
# reset env fitnesses
for envName in allEnvNames:
    envFitnesses[envName] = 0

if options.envGen > 0:
    with open('tpg-trainer-ggp.pkl','rb') as f:
        trainer = pickle.load(f)
else:
	trainer = TpgTrainer(actions=range(18), teamPopSizeInit=150)

processes = 40
pool = mp.Pool(processes=processes, initializer=limit_cpu, maxtasksperchild=5)
man = mp.Manager()

envPopSize = 9 # number of envs to up in envNamePop
envGapSize = 3 # number of envs to replace in envPop

numEpisodes = 5 # times to evaluate each env
numFrames = 200 # number of frames per episode, to increase as time goes on

allScores = [] # track all scores each generation

tStart = time.time()

envGen = options.envGen # generation of cycling through env pop

# create log file and header
logFileName = 'ggp-log-gens-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + '.txt'
with open(logFileName, 'a') as f:
    f.write('tpgGen,envGen,frames,envName,tpgMin,tpgMax,tpgAvg,envFit\n')
logFileMpName = 'ggp-log-multiperformance-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + '.txt'
with open(logFileMpName, 'a') as f:
    f.write('tpgGen,envGen,trainFrames,combined')
    for envName in allEnvNames:
        f.write(',' + envName)
    f.write('\n')
    
# get starting frames
if envGen >= 11 and envGen < 16:
    numFrames = 500
elif envGen >= 16 and envGen < 21:
    numFrames = 1000
elif envGen >= 21 and envGen < 26:
    numFrames = 2000
elif envGen >= 26 and envGen < 31:
    numFrames = 5000
elif envGen >= 31:
    numFrames = 18000
    
while True: # do generations with no end
    envGen += 1
    if envGen == 11:
        trainer.clearOutcomes()
        numFrames = 500
    elif envGen == 16:
        trainer.clearOutcomes()
        numFrames = 1000
    elif envGen == 21:
        trainer.clearOutcomes()
        numFrames = 2000
    elif envGen == 26:
        trainer.clearOutcomes()
        numFrames = 5000
    elif envGen == 31:
        trainer.clearOutcomes()
        numFrames = 18000
    
    # choose the new env name pop
    sortedEnvFits = sorted(envFitnesses.items(), key=operator.itemgetter(1), reverse=True)
    envNamesPop = [envFit[0] for envFit in sortedEnvFits[:envPopSize-envGapSize]] # keep top ones
    sortedNewEnvFits = sortedEnvFits[envPopSize:]
    random.shuffle(sortedNewEnvFits)
    for i in range(envGapSize): # replace gap size with random
        envNamesPop.append(sortedNewEnvFits[i][0])
        
    # reset env fitnesses
    for envName in allEnvNames:
        envFitnesses[envName] = 0
        
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Env Gen: ' + str(envGen))
    print('Envs: ' + str(envNamesPop))
    
    # run each env multiple times
    for ep in range(numEpisodes):
        print('On to Episode: ' + str(ep))
        # choose order of envs to run
        random.shuffle(envNamesPop)
        
        # run each env per episode
        for envName in envNamesPop:
            print('On to Game: ' + envName)
            
            scoreList = man.list() # reset score list
            # run the agents in the env
            pool.map(runAgent, 
                [(agent, envName, scoreList, 1, numFrames)
                for agent in trainer.getAllAgents(skipTasks=[envName],noRef=True)])
        
            trainer.applyScores(scoreList)
            trainer.evolve(fitShare=False, tasks=[envName], elitistTasks=allEnvNames)
            
            # save model after every gen
            with open('tpg-trainer-ggp.pkl','wb') as f:
                pickle.dump(trainer,f)
            
            tpgBest = trainer.scoreStats['max']
            tpgWorst = trainer.scoreStats['min']
            
            # scores of tpg agents normalized between 0 and 1
            if tpgBest != tpgWorst:
                tpgScores = [(score-tpgWorst)/(tpgBest-tpgWorst) for score in trainer.scoreStats['scores']]
            else:
                tpgScores = [0]*len(trainer.scoreStats['scores'])
            
            # calculate fitness of the environments for each agent
            tpgEnvFits = [(2*score/0.75)-(score)**2 for score in tpgScores]
            
            # the final fitness of the current environment
            envFit = sum(tpgEnvFits)/len(tpgEnvFits)
            
            # add score to fitness for environment
            envFitnesses[envName] += envFit
            
            # report to log
            with open(logFileName, 'a') as f:
                f.write(str(trainer.curGen) + ','
                    + str(envGen) + ','
                    + str(numFrames) + ','
                    + envName + ','
                    + str(trainer.scoreStats['min']) + ','
                    + str(trainer.scoreStats['max']) + ','
                    + str(trainer.scoreStats['average']) +  ','
                    + str(envFit) + '\n')
            
    # check how agents do on all titles
    if trainer.curGen % 1000 == 0:
        print('Evaluating agents on all envs.')
        scoreList = man.list() # reset score list
        agents = trainer.getBestAgents(tasks=allEnvNames, amount=5, topn=3)
        for envName in allEnvNames:
            pool.map(runAgent, 
                [(agent, envName, scoreList, 30, 18000)
                for agent in agents])
        
        
        for team in trainer.rootTeams:
	        if len(team.outcomes) == len(allEnvNames):
		    with open(logFileMpName, 'a') as f:
		        f.write(str(trainer.curGen) + ','
		            + str(envGen) + ','
		            + str(numFrames))
		        for envName in allEnvNames:
		            f.write(',' + str(team.outcomes[envName]))
		        f.write('\n')
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
