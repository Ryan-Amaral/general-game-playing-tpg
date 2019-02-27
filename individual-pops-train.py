"""
This file is used to train and save a trainer that has separate populations trained
on each game.
"""

from __future__ import division
from tpg.tpg_trainer import TpgTrainer
from tpg.tpg_agent import TpgAgent
from tpg_helpers import *
import random
import multiprocessing as mp
import pickle
import time

allEnvNames = ['Alien-v0','Asteroids-v0','Atlantis-v0','BankHeist-v0',
               'BattleZone-v0','Bowling-v0','Boxing-v0','Centipede-v0']

# create TPG
trainer = TpgTrainer(actions=range(18), teamPopSize=200, maxProgramSize=128, singlePop=False)

tstart = time.time()

# create populations
for envName in allEnvNames:
    trainer.createNewPopulation(popName=envName)

logName = 'sgp-log-8-pops.txt'
with open(logName, 'a') as f:
    f.write('tpgGen,hoursElapsed,envName,tpgMin,tpgMax,tpgAvg,eliteSize,eliteUid\n')

while True: # train indefinately
    print('TPG Gen: ' + str(trainer.populations[envName].curGen))
    for envName in allEnvNames: # train on each env
        print('Playing Game: ' + envName)

        scoreList = man.list()

        # run all agents on env
        pool.map(runAgent,
            [(agent, envName, scoreList, trainEpisodes, trainFrames, None)
                for agent in trainer.getAllAgents(skipTasks=[envName], noRef=True,
                        popName=envName)])

        trainer.applyScores(scoreList, popName=envName)

        # report curEnv results to log
        scoreStats = trainer.getTaskScores(envName, popName=envName)
        bestTeam = trainer.getBestTeams(tasks=[envName], popName=envName)[0]
        with open(logName, 'a') as f:
            f.write(str(trainer.populations[envName].curGen) + ','
                + str((time.time()-tstart)/3600) + ','
                + envName + ','
                + str(scoreStats['min']) + ','
                + str(scoreStats['max']) + ','
                + str(scoreStats['average']) +  ','
                + str(len(bestTeam.getRootTeamGraph()[0])) + ','
                + str(bestTeam.uid) + '\n')

        # do evolution on each env played
        trainer.evolve(fitMthd='single', tasks=[envName], elitistTasks=[envName], popName=envName)

# save model at end
with open('trainer-8-pops.pkl','wb') as f:
    pickle.dump(trainer,f)
