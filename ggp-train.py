"""
General Game Playing script. Trains a Tangled Program Graph model on multiple games.
Created by: Ryan Amaral
Created on: Feb. 25, 2019 (based on much earlier version)
"""

from __future__ import division
from tpg.tpg_trainer import TpgTrainer
from tpg.tpg_agent import TpgAgent
from optparse import OptionParser
from tpg_helpers import *
import random
import datetime

"""
Command line arguments.
"""

parser = OptionParser()
parser.add_option('-g', '--games', type='int', dest='numGames', default=16) # number of games to play
parser.add_option('-r', '--randg', action='store_true', dest='randomGames', default=False) # choose games randomly
parser.add_option('-p', '--pop', type='int', dest='popSize', default=600) # starting tpg population size
parser.add_option('-m', '--popMax', type='int', dest='popSizeMax', default=600) # population size to work up to
parser.add_option('-w', '--workers', type='int', dest='workers', default=3) # concurrent workers for parallization
parser.add_option('-t', '--type', type='int', dest='trainType', default=0) # method of training
# trainType is 0 for 'all at once', 1 for 'merge', 2 for 'gradual merge'.

(options, args) = parser.parse_args()

"""
Create and initialize log files.
"""

timeStamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))

# game score summary of each game in each generation
logFileGameScoresName = 'ggp-log-gamescores-' + timeStamp + '.txt'
if options.envGen == 0:
    with open(logFileGenName, 'a') as f:
        f.write('tpgGen,envName,tpgMin,tpgMax,tpgAvg,eliteSize,eliteUid\n')

# every few generations evaluate top overall few teams on all titles
logFileChampionsName = 'ggp-log-champions-' + timeStamp + '.txt'
if options.envGen == 0:
    with open(logFileMpName, 'a') as f:
        f.write('tpgGen,teamId')
        for envName in allEnvNames:
            f.write(',score' + envName)
        for envName in allEnvNames:
            f.write(',vis' + envName)
        f.write(',visTotal\n')

# population fitness summary each generation
logFileFitnessName = 'ggp-log-fitness-' + options.timeStamp + '.txt'
if options.envGen == 0:
    with open(logFileFitName, 'a') as f:
        f.write('tpgGen,envs,fitMin,fitMax,fitAvg,champSize,champUid,totalTeams,totalRootTeams\n')

# and create file to save TPG
trainerFileName = 'tpg-trainer-' + options.timeStamp + '.pkl'

"""
General setup for GGP.
"""


# all of the titles we will be general game playing on
# we chose games that we know TPG does at-least OK in alone
allEnvNames = ['Alien-v0','Asteroids-v0','Atlantis-v0','BankHeist-v0',
               'BattleZone-v0','Bowling-v0','Boxing-v0','Centipede-v0',
               'ChopperCommand-v0','DoubleDunk-v0','FishingDerby-v0',
               'Freeway-v0','Frostbite-v0','Gravitar-v0','Hero-v0',
               'IceHockey-v0','Jamesbond-v0','Kangaroo-v0','Krull-v0',
               'KungFuMaster-v0','MsPacman-v0','PrivateEye-v0',
               'RoadRunner-v0','Skiing-v0','Tennis-v0','TimePilot-v0',
               'UpNDown-v0','Venture-v0','WizardOfWor-v0','Zaxxon-v0']

# shuffle if random and take certain number of games
if options.randomGames:
    random.shuffle(allEnvNames)
allEnvNames = allEnvNames[:numGames]

pool = mp.Pool(processes=options.workers, initializer=limit_cpu, maxtasksperchild=5)
man = mp.Manager() # manager for shared memory lists

"""
Method for training TPG on all games at once. Each individual in the single population
will see all of the games before evolution will occur.
"""
def ggpTrainAllAtOnce(envNames, popSize, lfGSName, lfCName, lfFName, trainerFileName, pool, man,
        trainFrames=18000, testFrames=18000, trainEpisodes=3, evalEpisodes=10):
    # create TPG
    trainer = TpgTrainer(actions=range(18), teamPopSize=popSize, maxProgramSize=128)

    envNamesSrt = sorted(list(envNames)) # for reporting envs played

    while True: # train indefinately
        print('TPG Gen: ' + str(trainer.curGen))
        random.shuffle(envNames) # I don't think this actually matters
        for envName in envNames: # train on each env
            print('Playing Game: ' + envName)

            scoreList = man.list()

            # run all agents on env
            pool.map(runAgent,
                [(agent, envName, scoreList, trainEpisodes, trainFrames, None)
                    for agent in trainer.getAllAgents(skipTasks=[envName], noRef=True)])

            trainer.applyScores(scoreList)

            # report curEnv results to log
            scoreStats = trainer.getTaskScores(envName)
            with open(lfGSName, 'a') as f:
                f.write(str(trainer.curGen) + ','
                    + envName + ','
                    + str(scoreStats['min']) + ','
                    + str(scoreStats['max']) + ','
                    + str(scoreStats['average']) +  ','
                    + str(len(trainer.getBestAgents(tasks=[envName],
                                amount=1,topn=1)[0].team.getRootTeamGraph()[0])) + ','
                    + str(trainer.getBestAgents(tasks=[envName],
                                amount=1,topn=1)[0].getUid()) + '\n')

        # do evolution after all envs played # update combine in tpg first!!!!!!!!!!!!!!!!!!!!!!!!!
        trainer.evolve(fitMthd='combine', tasks=envNames, elitistTasks=envNames)

        # report generational fitness results
        ### Must fix getting top agent!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        with open(lfFName, 'a') as f:
            f.write(str(trainer.curGen) + ','
                + '/'.join(envNamesSrt) + ','
                + str(trainer.scoreStats['min']) + ','
                + str(trainer.scoreStats['max']) + ','
                + str(trainer.scoreStats['average']) +  ','
                + str(len(trainer.getBestAgents(tasks=[envName],amount=1,topn=1)[0].team.getRootTeamGraph()[0])) + ','
                + str(len(trainer.teams)) + ','
                + str(len(trainer.rootTeams)) + ','
                + str(trainer.getBestAgents(tasks=[envName],amount=1,topn=1)[0].getUid()) + '\n')

        # save model after every gen
        with open(trainerFile,'wb') as f:
            pickle.dump(trainer,f)

        # every 50 generations evaluate top agents on all games
        if trainer.curGen % 50 == 0:
            champEval()


"""
Method for training TPG on each game separately, in separate populations. All of
the populations will be merged together at once.
"""
def ggpTrainMerge(envNames, popSize, popSizeMax, lfGSName, lfCName, lfFName,
        pool, man):
    pass

"""
Method for training TPG on each game separately, in separate populations. The populations
will be merged gradually, meaning that first separate populations will be paired,
then those pairs will be paired, and so on untill in one population.
"""
def ggpTrainGradualMerge(envNames, popSize, popSizeMax, lfGSName, lfCName, lfFName,
        pool, man):
    pass

"""
Run the training unending.
"""

if options.trainType == 0:
    ggpTrainAllAtOnce(allEnvNames, options.popSize, logFileGameScoresName,
        logFileChampionsName, logFileFitnessName, pool, man)
elif options.trainType == 1:
    ggpTrainMerge(allEnvNames, options.popSize, options.popSizeMax, logFileGameScoresName,
        logFileChampionsName, logFileFitnessName, pool, man)
else:
    ggpTrainGradualMerge(allEnvNames, options.popSize, options.popSizeMax, logFileGameScoresName,
        logFileChampionsName, logFileFitnessName, pool, man)
