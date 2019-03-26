"""
Script for testing performance profile of TPG runs.
Created by: Ryan Amaral
Created on: Mar. 23, 2019
"""

from __future__ import division
from tpg.tpg_trainer import TpgTrainer
from tpg.tpg_agent import TpgAgent
from tpg_helpers import *
import random
import datetime
import multiprocessing as mp
from timeit import default_timer as timer
import cProfile
import gym
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-p', '--pop', type='int', dest='popSize', default=200)
# concurrent workers for parallization
parser.add_option('-w', '--workers', type='int', dest='workers', default=3)
# number of frames to train on
parser.add_option('--trainFrames', type='int', dest='trainFrames', default=18000)
# number of episodes to train on
parser.add_option('--trainEps', type='int', dest='trainEps', default=3)
parser.add_option('--env', type='string', dest='envName', default='Boxing-v0')
parser.add_option('-g', type='int', dest='generations', default=300)


timeStamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
# write to log with cProfile.run(func(), filtename=logFile)
logFileTime = 'time-profile-log-' + timeStamp + '.txt'
with open(logFile, 'a') as f:
    f.write('popSize: ' + str(options.popSize) + '\n')
    f.write('envName: ' + options.envName + '\n')
    f.write('trainEps: ' + str(options.trainEps) + '\n')
    f.write('trainFrames: ' + str(options.trainFrames) + '\n')
    f.write('Generations: ' + str(options.generations) + '\n\n')

# game score summary of each game in each generation
logFileTpg = 'tpg-log-' + timeStamp + '.txt'
with open(logFileTpg, 'a') as f:
    f.write('tpgGen,hoursElapsed,env,fitMin,fitMax,fitAvg,champSize,champUid,totalTeams,totalRootTeams\n')

pool = mp.Pool(processes=options.workers, initializer=limit_cpu, maxtasksperchild=1)
man = mp.Manager() # manager for shared memory lists

def runTpg():

    tmpEnv = gym.make(options.envName)
    # create TPG
    trainer = TpgTrainer(actions=range(tmpEnv.action_space.n),
        teamPopSize=options.popSize, rTeamPopSize=options.popSize,
        maxProgramSize=128)

    tmpEnv.close()

    print('Playing Game: ' + options.envName)

    while train.populations[None].curGen < options.generations: # train indefinately
        print('TPG Gen: ' + str(trainer.populations[None].curGen))

        scoreList = man.list()

        # run all agents on env
        pool.map(runAgent,
            [(agent, options.envName, scoreList, options.trainEps, options.trainFrames, None)
                for agent in trainer.getAllAgents(skipTasks=[options.envName], noRef=True)])

        trainer.applyScores(scoreList)

        # do evolution after all envs played
        trainer.evolve(tasks=[options.envName], elitistTasks=[options.envName])

        # report generational fitness results
        bestTeam = trainer.getBestTeams(tasks=[options.envName])[0]
        with open(logFileTpg, 'a') as f:
            f.write(str(trainer.populations[None].curGen) + ','
                + str((time.time()-tstart)/3600) + ','
                + options.envName + ','
                + str(trainer.populations[None].scoreStats['min']) + ','
                + str(trainer.populations[None].scoreStats['max']) + ','
                + str(trainer.populations[None].scoreStats['average']) +  ','
                + str(len(bestTeam.getRootTeamGraph()[0])) + ','
                + str(bestTeam.uid) + ','
                + str(len(trainer.populations[None].teams)) + ','
                + str(len(trainer.populations[None].rootTeams)) + '\n')

cProfile.run(runTpg, logFileTime)
