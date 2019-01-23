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
import numpy as np

parser = OptionParser()
parser.add_option('-b', '--big', action='store_true', dest='bigSetOGames', default=False) # whether to use all 30 titles or just 15
parser.add_option('-g', '--envgen', type='int', dest='envGen', default=0) # env gen to continue from when restarting run
parser.add_option('-t', '--timeStamp', type='str', dest='timeStamp', default=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) # timestamp of run to continue, or new timestamp
parser.add_option('-p', '--pop', type='int', dest='popSize', default=750) # tpg population size
parser.add_option('-s', '--shrink', action='store_true', dest='envPopShrink', default=False) # whether to shrink title population from max size, or keep constant
parser.add_option('-z', '--eps', type='int', dest='envPopSize', default=9) # environment population size for when continuing a run
parser.add_option('-v', '--vir', action='store_true', dest='virulence', default=False) # whether to use host parasite virulence idea for title fitness
parser.add_option('-c', '--cores', type='int', dest='cores', default=3) # number of cpu cores to use for training
(options, args) = parser.parse_args()

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
if not options.bigSetOGames:
    allEnvNames = allEnvNames[:15]

envFitnesses = {}
# reset env fitnesses
for envName in allEnvNames:
    envFitnesses[envName] = 0

trainerFileName = 'tpg-trainer-ggp-' + options.timeStamp + '.pkl'

if options.envGen > 0:
    with open(trainerFileName, 'rb') as f:
        trainer = pickle.load(f)
else:
    trainer = TpgTrainer(actions=range(18), teamPopSize=options.popSize, maxProgramSize=128)

envPopSize = 9 # number of envs to up in envNamePop
envGapSize = 3 # number of envs to replace in envPop

if options.envPopShrink: # start it on all games
    if options.envGen > 0: # through some generations so get right size
        envPopSize = options.eps
    else: # just starting so set to all games
        envPopSize = len(allEnvNames)

numEpisodes = 5 # times to evaluate each env
numFrames = 500 # number of frames per episode, to increase as time goes on

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
    f.write('tpgGen,envGen,gamesPlayed,frames,envName,tpgMin,tpgMax,tpgAvg,envFit,championSize,popsize,totalTeams,totalRootTeams,champUid\n')

logFileMpName = 'ggp-log-multiperf-' + options.timeStamp + '.txt'
with open(logFileMpName, 'a') as f:
    f.write('tpgGen,envGen,trainFrames,teamId')
    for envName in allEnvNames:
        f.write(',score' + envName)
    for envName in allEnvNames:
        f.write(',vis' + envName)
    f.write(',visTotal\n')

# get starting frames
if trainer.curGen >= 50 and trainer.curGen < 100:
    numFrames = 1000
elif trainer.curGen >= 100 and trainer.curGen < 150:
    numFrames = 2000
elif trainer.curGen >= 150 and trainer.curGen < 200:
    numFrames = 5000
elif trainer.curGen >= 200 and trainer.curGen < 250:
    numFrames = 10000
elif trainer.curGen >= 250:
    numFrames = 18000

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
        numRandFrames = random.randint(1,min(20, numFrames))
        for i in range(numFrames): # frame loop
            if i < numRandFrames:
                _, _, isDone, _ = env.step(env.action_space.sample())
                if isDone: # don't count it if lose on random steps
                    rep -= 1
                continue

            act = agent.act(getState(np.array(state, dtype=np.int32)), valActs=valActs)

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
        numRandFrames = random.randint(1,min(20, numFrames))
        for i in range(numFrames): # frame loop
            if i < numRandFrames:
                _, _, isDone, _ = env.step(env.action_space.sample())
                if isDone: # don't count it if lose on random steps
                    rep -= 1
                continue

            act = agent.act(getState(np.array(state, dtype=np.int32)), valActs=valActs)

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
# probably won't even be used
def limit_cpu():
    p = psutil.Process(os.getpid())
    p.nice(0)

# testing some of the top agents on all titles
def multiTest():
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

processes = options.cores
pool = mp.Pool(processes=processes, initializer=limit_cpu, maxtasksperchild=5)
man = mp.Manager()

allScores = [] # track all scores each generation

tStart = time.time()

envGen = options.envGen # continue onto a certain generation

gamesPlayed = 0 # track total games attempted by tpg

tasksToSkip = []

# main training loop
while True: # do generations with no end
    envGen += 1

    # choose the new env name pop
    if options.virulence:
        sortedEnvFits = sorted(envFitnesses.items(), key=operator.itemgetter(1), reverse=True)
    else:
        sortedEnvFits = sorted(envFitnesses.items(), key=lambda x: random.random())
    envNamesPop = [envFit[0] for envFit in sortedEnvFits[:envPopSize-envGapSize]] # keep top ones
    sortedNewEnvFits = sortedEnvFits[envPopSize-envGapSize:]
    random.shuffle(sortedNewEnvFits)
    for i in range(min(envGapSize, len(sortedNewEnvFits))): # replace gap size with random
        envNamesPop.append(sortedNewEnvFits[i][0])

    if options.envPopShrink and envPopSize > 9:
        envPopSize -= envGapSize

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

        agents = trainer.getAllAgents(skipTasks=tasksToSkip, noRef=True)

        # run each env per episode
        for envName in envNamesPop:
            if envName not in tasksToSkip:
                tasksToSkip.append(envName) # can skip now

            print('On to Game: ' + envName)

            scoreList = man.list() # reset score list

            # run the agents in the env
            pool.map(runAgent,
                [(agent, envName, scoreList, 1, numFrames)
                for agent in agents])

            gamesPlayed += 1

            # update agents in trainer
            agents = [TpgAgent(team) for team in trainer.applyScores(scoreList)]

            scoreStats = trainer.getTaskScores(envName)

            tpgBest = scoreStats['max']
            tpgWorst = scoreStats['min']

            # scores of tpg agents normalized between 0 and 1
            if tpgBest != tpgWorst:
                tpgScores = [(score-tpgWorst)/(tpgBest-tpgWorst) for score in scoreStats['scores']]
            else:
                tpgScores = [0]*len(scoreStats['scores'])

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
                    + str(gamesPlayed) + ','
                    + str(numFrames) + ','
                    + envName + ','
                    + str(scoreStats['min']) + ','
                    + str(scoreStats['max']) + ','
                    + str(scoreStats['average']) +  ','
                    + str(envFit) + ','
                    + str(len(trainer.getBestAgents(tasks=[envName],amount=1,topn=1)[0].team.getRootTeamGraph()[0])) + ','
                    + str(len(trainer.teams)) + ','
                    + str(len(trainer.rootTeams)) + ',',
                    + str(trainer.getBestAgents(tasks=[envName],amount=1,topn=1)[0].getUid()) + '\n')

        trainer.evolve(fitMthd='combine', tasks=[envNamesPop], elitistTasks=allEnvNames)

        # save model after every gen
        with open(trainerFileName,'wb') as f:
            pickle.dump(trainer,f)

        # do multitest every 60 generations
        if(trainer.curGen % 60 == 0):
            multiTest()

        # update training frames for future generations
        # and make tasks unskippable to start
        if trainer.curGen == 50:
            numFrames = 1000
            tasksToSkip = []
        elif trainer.curGen == 100:
            numFrames = 2000
            tasksToSkip = []
        elif trainer.curGen == 150:
            numFrames = 5000
            tasksToSkip = []
        elif trainer.curGen == 200:
            numFrames = 10000
            tasksToSkip = []
        elif trainer.curGen == 250:
            numFrames = 18000
            tasksToSkip = []
