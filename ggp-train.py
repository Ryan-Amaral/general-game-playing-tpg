# imports and helper methods
from __future__ import division
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
parser.add_option('-g', '--envgen', type='int', dest='envGen', default=0)
parser.add_option('-t', '--timeStamp', type='str', dest='timeStamp', default=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
parser.add_option('-p', '--pop', type='int', dest='popSize', default=150)
parser.add_option('-s', '--shrink', action='store_false', dest='envPopShrink', default=False)
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
            outState[cnt] = (((inState[row][col][0] >> 2) << 12)
                          + ((inState[row][col][1] >> 2) << 6)
                          + (inState[row][col][2] >> 2)) # to get RRRRRR GGGGGG BBBBBB
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
    
    env = gym.make(envName)
    valActs = range(env.action_space.n) # valid actions, some envs are less
    
    scoreTotal = 0 # score accumulates over all episodes
    for rep in range(numRepeats): # episode loop
        state = env.reset()
        scoreEp = 0
        numRandFrames = 0
        if numRepeats > 1:
            numRandFrames = random.randint(1,30)
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
    
"""
This one used for tracking visuals indexed
Args:
    args: (TpgAgent, envName, scoreList, numRepeats, numFrames)
"""
def runAgent2(args):
    agent = args[0]
    envName = args[1]
    scoreList = args[2]
    numRepeats = args[3] # number of times to repeat game
    numFrames = args[4]
    visTrack = args[5]
    
    env = gym.make(envName)
    valActs = range(env.action_space.n) # valid actions, some envs are less
    
    scoreTotal = 0 # score accumulates over all episodes
    for rep in range(numRepeats): # episode loop
        state = env.reset()
        scoreEp = 0
        numRandFrames = 0
        if numRepeats > 1:
            numRandFrames = random.randint(1,30)
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
    visTrack[agent.team.uid] = agent.screenIndexed
    
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

todoEnvNames = list(allEnvNames)

envFitnesses = {}
# reset env fitnesses
for envName in allEnvNames:
    envFitnesses[envName] = 0

trainerFileName = 'tpg-trainer-ggp-' + options.timeStamp + '.pkl'

if options.envGen > 0:
    with open(trainerFileName, 'rb') as f:
        trainer = pickle.load(f)
else:
    trainer = TpgTrainer(actions=range(18), teamPopSizeInit=options.popSize)

processes = 40
pool = mp.Pool(processes=processes, initializer=limit_cpu, maxtasksperchild=5)
man = mp.Manager()

envPopSize = 9 # number of envs to up in envNamePop
envGapSize = 3 # number of envs to replace in envPop

envPopShrink = options.envPopShrink # whether to start at all games, and shrink down

if envPopShrink: # start it big
    envPopSize = len(allEnvNames)

numEpisodes = 5 # times to evaluate each env
numFrames = 200 # number of frames per episode, to increase as time goes on

allScores = [] # track all scores each generation

tStart = time.time()

envGen = options.envGen # generation of cycling through env pop

# create log files and headers
logFilePosName = 'ggp-log-pos-' + options.timeStamp + '.txt'
with open(logFilePosName, 'a') as f:
    f.write('tpgGen,envGen,trainFrames')
    for envName in allEnvNames:
        f.write(',1st-' + envName)
    for envName in allEnvNames:
        f.write(',2nd-' + envName)
    for envName in allEnvNames:
        f.write(',3rd-' + envName)
    for envName in allEnvNames:
        f.write(',4th-' + envName)    
    for envName in allEnvNames:
        f.write(',5th-' + envName)
    f.write('\n')

logFileGenName = 'ggp-log-gens-' + options.timeStamp + '.txt'
with open(logFileGenName, 'a') as f:
    f.write('tpgGen,envGen,frames,envName,tpgMin,tpgMax,tpgAvg,envFit,championSize,popsize,totalTeams,totalRootTeams\n')
    
logFileMpName = 'ggp-log-multiperf-' + options.timeStamp + '.txt'
with open(logFileMpName, 'a') as f:
    f.write('tpgGen,envGen,trainFrames,teamId')
    for envName in allEnvNames:
        f.write(',score' + envName)
    for envName in allEnvNames:
        f.write(',vis' + envName)
    f.write(',visTotal\n')
    
# get starting frames
if envGen >= 6 and envGen < 11:
    numFrames = 500
elif envGen >= 11 and envGen < 16:
    numFrames = 1000
elif envGen >= 16 and envGen < 21:
    numFrames = 2000
elif envGen >= 21 and envGen < 31:
    numFrames = 5000
elif envGen >= 31:
    numFrames = 18000
    
multiTest = False # flag for whether to to big test on all games for best few agents
    
while True: # do generations with no end
    envGen += 1
    if envGen == 6:
        todoEnvNames = list(allEnvNames)
        numFrames = 500
    elif envGen == 11:
        todoEnvNames = list(allEnvNames)
        numFrames = 1000
    elif envGen == 16:
        todoEnvNames = list(allEnvNames)
        numFrames = 2000
    elif envGen == 21:
        todoEnvNames = list(allEnvNames)
        numFrames = 5000
    elif envGen == 31:
        todoEnvNames = list(allEnvNames)
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
        print('Tpg Gen: ' + str(trainer.curGen))
        print('On to Episode: ' + str(ep))
        # choose order of envs to run
        random.shuffle(envNamesPop)
        
        # run each env per episode
        for envName in envNamesPop:
            print('On to Game: ' + envName)
            
            scoreList = man.list() # reset score list
            
            # skip task/env for some agents if already done with it
            if envName in todoEnvNames:
                skipTask = 'none'
                if ep == numEpisodes - 1:
                    todoEnvNames.remove(envName)
            else:
                skipTask = envName
                
            # run the agents in the env
            pool.map(runAgent, 
                [(agent, envName, scoreList, 1, numFrames)
                for agent in trainer.getAllAgents(skipTasks=[skipTask], noRef=True)])
                
            trainer.applyScores(scoreList)
            trainer.evolve(fitShare=False, tasks=[envName], elitistTasks=allEnvNames)
            
            # save model after every gen
            with open(trainerFileName,'wb') as f:
                pickle.dump(trainer,f)
            
            tpgBest = trainer.scoreStats['max']
            tpgWorst = trainer.scoreStats['min']
            
            # scores of tpg agents normalized between 0 and 1
            if tpgBest != tpgWorst:
                tpgScores = [(score-tpgWorst)/(tpgBest-tpgWorst) for score in trainer.scoreStats['scores']]
            else:
                tpgScores = [0]*len(trainer.scoreStats['scores'])
            
            # calculate fitness of the environments for each agent
            tpgEnvFits = [(2*score/0.75)-(score/0.75)**2 for score in tpgScores]
            
            # the final fitness of the current environment
            envFit = sum(tpgEnvFits)/len(tpgEnvFits)
            
            # add score to fitness for environment
            envFitnesses[envName] += envFit
            
            # report to log
            with open(logFileGenName, 'a') as f:
                f.write(str(trainer.curGen) + ','
                    + str(envGen) + ','
                    + str(numFrames) + ','
                    + envName + ','
                    + str(trainer.scoreStats['min']) + ','
                    + str(trainer.scoreStats['max']) + ','
                    + str(trainer.scoreStats['average']) +  ','
                    + str(envFit) + ','
                    + str(len(trainer.getBestAgents(tasks=[envName],amount=1,topn=1)[0].team.getRootTeamGraph()[0])) 
                    + str(len(trainer.teams)) + ','
                    + str(len(trainer.rootTeams)) + '\n')
            
    if (envGen <= 30 and envGen % 15 == 0) or (envGen > 30 and envGen % 10 == 0):
        multiTest = True
            
    # check how agents do on all titles
    if multiTest:
        multiTest = False
        print('Evaluating agents on all envs.')
        agents = trainer.getBestAgents(tasks=allEnvNames, amount=5, topn=5)
        
        agentsPos = trainer.getAgentsPositions(tasks=allEnvNames, topn=5)
        # log the positions
        with open(logFilePosName, 'a') as f:
            f.write(str(trainer.curGen) + ','
                        + str(envGen) + ','
                        + str(numFrames))
            for pos in range(5):
                for envName in allEnvNames:
                    if len(agentsPos[envName]) > pos:
                        val = agentsPos[envName][pos].team.uid
                    else:
                        val = -1
                    f.write(',' + str(val))
            f.write('\n')
        
        agentScores = {} # save scores of agents here
        allVisTrack = {} # save indexed screen space here
        for tId in [agent.team.uid for agent in agents]:
            agentScores[tId] = {}
            allVisTrack[tId] = {}
        for envName in allEnvNames:
            scoreList = man.list() # reset score list
            visTrack = man.dict()
            pool.map(runAgent2, 
                [(agent, envName, scoreList, 30, 18000, visTrack)
                for agent in agents])
            # put scores in dict for env
            for score in scoreList:
                agentScores[score[0]][envName] = score[1][envName]
            for uid in visTrack.keys():
                allVisTrack[uid][envName] = visTrack[uid]
                
        for uid in allVisTrack:
            allVisTrack[uid]['visTotal'] = [0]*len(allVisTrack[uid][allEnvNames[0]])
            for envName in allEnvNames:
                for i in range(len(allVisTrack[uid][envName])):
                    if allVisTrack[uid][envName][i] == 1 and i < len(allVisTrack[uid]['visTotal']):
                        allVisTrack[uid]['visTotal'][i] = 1
        
        with open(logFileMpName, 'a') as f:
            for uid in agentScores:
                f.write(str(trainer.curGen) + ','
                        + str(envGen) + ','
                        + str(numFrames) + ','
                        + str(uid))
                for envName in allEnvNames:
                    f.write(',' + str(agentScores[uid][envName]))
                for envName in allEnvNames:
                    f.write(',' + str(allVisTrack[uid][envName].count(1) / len(allVisTrack[uid][envName])))
                f.write(',' + str(allVisTrack[uid]['visTotal'].count(1) / len(allVisTrack[uid]['visTotal'])) + '\n')
            
            
