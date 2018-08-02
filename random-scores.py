# imports and helper methods
import gym
import gym.spaces

import multiprocessing as mp
import psutil

"""
Run each agent in this method for parallization.
Args:
    args: (TpgAgent, envName, scoreList, numEpisodes, numFrames)
"""
def runAgent(envName):
    env = gym.make(envName)
    scoreTotal = 0 # score accumulates over all episodes
    for ep in range(20): # episode loop
        state = env.reset()
        scoreEp = 0
        for i in range(1000): # frame loop
            state, reward, isDone, debug = env.step(env.action_space.sample())
            scoreEp += reward # accumulate reward in score
            if isDone:
                break # end early if losing state
        scoreTotal += scoreEp

    print(str(env.env) + ': ' + str(scoreTotal/20))
       
    env.close()
    
# https://stackoverflow.com/questions/42103367/limit-total-cpu-usage-in-python-multiprocessing/42130713
def limit_cpu():
    p = psutil.Process(os.getpid())
    p.nice(10)
    
# all of the titles we will be general game playing on
# we chose games that we know TPG does OK in alone
envNames = ['Alien-v0','Asteroids-v0','Atlantis-v0','BankHeist-v0',
               'BattleZone-v0','Bowling-v0','Boxing-v0','Centipede-v0',
               'ChopperCommand-v0','DoubleDunk-v0','FishingDerby-v0',
               'Freeway-v0','Frostbite-v0','Gravitar-v0','Hero-v0',
               'IceHockey-v0','Jamesbond-v0','Kangaroo-v0','Krull-v0',
               'KungFuMaster-v0','MsPacman-v0','PrivateEye-v0',
               'RoadRunner-v0','Tennis-v0','TimePilot-v0',
               'UpNDown-v0','Venture-v0','WizardOfWor-v0','Zaxxon-v0']

processes = 2
pool = mp.Pool(processes=processes, initializer=limit_cpu)

  
pool.map(runAgent, [envName for envName in envNames])
 

