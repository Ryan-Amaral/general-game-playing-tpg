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
import multiprocessing as mp
import pickle
import time

"""
Command line arguments.
"""

def separgs(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

parser = OptionParser()
# number of games to play
parser.add_option('-n', '--ngames', type='int', dest='numGames', default=16)
# specify any games
parser.add_option('--games', type='string', action='callback', callback=separgs, dest='games')
# choose games randomly
parser.add_option('-r', '--randg', action='store_true', dest='randomGames', default=False)
# starting tpg population size
parser.add_option('-p', '--pop', type='int', dest='popSize', default=360)
# population size to work up to
parser.add_option('-m', '--popMax', type='int', dest='popSizeMax', default=360)
# concurrent workers for parallization
parser.add_option('-w', '--workers', type='int', dest='workers', default=3)
# method of training
parser.add_option('-t', '--type', type='int', dest='trainType', default=0)
# number of frames to train on
parser.add_option('--trainFrames', type='int', dest='trainFrames', default=18000)
# number of frames to test on
parser.add_option('--testFrames', type='int', dest='testFrames', default=18000)
# number of episodes to train on
parser.add_option('--trainEps', type='int', dest='trainEps', default=3)
# number of episodes to train on
parser.add_option('--testEps', type='int', dest='testEps', default=10)
# every how many generations to do champ eval
parser.add_option('-c', '--champEvalGen', type='int', dest='champEvalGen', default=50)
# file for trainer with pretrained populations
parser.add_option('--popsFile', type='string', dest='popsFile', default=None)

# trainType is 0 for 'all at once', 1 for 'merge', 2 for 'gradual merge'.

(options, args) = parser.parse_args()

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

if options.games is not None:
    allEnvNames = sorted(options.games)
else:
    # shuffle if random and take certain number of games
    if options.randomGames:
        random.shuffle(allEnvNames)
    allEnvNames = sorted(allEnvNames[:options.numGames])
    #allEnvNames = ['Assault-v0', 'Boxing-v0'] # hardcode in games to use

print('All Games: ' + str(allEnvNames))

pool = mp.Pool(processes=options.workers, initializer=limit_cpu, maxtasksperchild=1)
man = mp.Manager() # manager for shared memory lists

"""
Create and initialize log files.
"""

timeStamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

# game score summary of each game in each generation
logFileGameScoresName = 'ggp-log-gamescores-' + timeStamp + '.txt'
with open(logFileGameScoresName, 'a') as f:
    f.write('tpgGen,hoursElapsed,envName,tpgMin,tpgMax,tpgAvg,eliteSize,eliteUid\n')

# every few generations evaluate top overall few teams on all titles
logFileChampionsName = 'ggp-log-champions-' + timeStamp + '.txt'
with open(logFileChampionsName, 'a') as f:
    f.write('tpgGen,hoursElapsed,teamId')
    for envName in allEnvNames:
        f.write(',score' + envName)
    for envName in allEnvNames:
        f.write(',vis' + envName)
    f.write(',visTotal\n')

# population fitness summary each generation
logFileFitnessName = 'ggp-log-fitness-' + timeStamp + '.txt'
with open(logFileFitnessName, 'a') as f:
    f.write('tpgGen,hoursElapsed,envs,fitMin,fitMax,fitAvg,champSize,champUid,totalTeams,totalRootTeams\n')

# and create file to save TPG
trainerFileName = 'tpg-trainer-' + timeStamp + '.pkl'

"""
Method for training TPG on all games at once. Each individual in the single population
will see all of the games before evolution will occur.
"""
def ggpTrainAllAtOnce():

    # create TPG
    trainer = TpgTrainer(actions=range(18), teamPopSize=options.popSize,
                    rTeamPopSize=options.popSize, maxProgramSize=128)

    envNamesSrt = sorted(list(allEnvNames)) # for reporting envs played

    while True: # train indefinately
        print('TPG Gen: ' + str(trainer.populations[None].curGen))
        for envName in allEnvNames: # train on each env
            print('Playing Game: ' + envName)

            scoreList = man.list()

            # run all agents on env
            pool.map(runAgent,
                [(agent, envName, scoreList, options.trainEps, options.trainFrames, None)
                    for agent in trainer.getAllAgents(skipTasks=[envName], noRef=True)])

            trainer.applyScores(scoreList)

            # report curEnv results to log
            scoreStats = trainer.getTaskScores(envName)
            bestTeam = trainer.getBestTeams(tasks=[envName])[0]
            with open(logFileGameScoresName, 'a') as f:
                f.write(str(trainer.populations[None].curGen) + ','
                    + str((time.time()-tstart)/3600) + ','
                    + envName + ','
                    + str(scoreStats['min']) + ','
                    + str(scoreStats['max']) + ','
                    + str(scoreStats['average']) +  ','
                    + str(len(bestTeam.getRootTeamGraph()[0])) + ','
                    + str(bestTeam.uid) + '\n')

        # do evolution after all envs played
        trainer.multiEvolve(tasks=[allEnvNames]+[[en] for en in allEnvNames],
                            weights=[0.5]+[0.5/len(allEnvNames) for _ in allEnvNames],
                            fitMethod='min', elitistTasks=allEnvNames)

        # report generational fitness results
        bestTeam = trainer.getBestTeams(tasks=envNamesSrt)[0]
        with open(logFileFitnessName, 'a') as f:
            f.write(str(trainer.populations[None].curGen) + ','
                + str((time.time()-tstart)/3600) + ','
                + '/'.join(envNamesSrt) + ','
                + str(trainer.populations[None].scoreStats['min']) + ','
                + str(trainer.populations[None].scoreStats['max']) + ','
                + str(trainer.populations[None].scoreStats['average']) +  ','
                + str(len(bestTeam.getRootTeamGraph()[0])) + ','
                + str(bestTeam.uid) + ','
                + str(len(trainer.populations[None].teams)) + ','
                + str(len(trainer.populations[None].rootTeams)) + '\n')

        # save model after every gen
        with open(trainerFileName,'wb') as f:
            pickle.dump(trainer,f)

        # every 50 generations evaluate top agents on all games
        if trainer.populations[None].curGen % options.champEvalGen == 0:
            champEval(envNamesSrt, trainer, logFileChampionsName, pool, man,
                        tstart, frames=options.testFrames, eps=options.testEps)


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
tstart = time.time() # start timing

if options.trainType == 0:
    ggpTrainAllAtOnce()
elif options.trainType == 1:
    ggpTrainMerge()
else:
    ggpTrainGradualMerge()
