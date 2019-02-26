# imports and helper methods
from __future__ import division
from tpg.tpg_trainer import TpgTrainer
from tpg.tpg_agent import TpgAgent
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-g', '--games', type='int', dest='numGames', default=16) # number of games to play
parser.add_option('-r', '--randg', action='store_true', dest='randomGames', default=False) # choose games randomly
parser.add_option('-p', '--pop', type='int', dest='popSize', default=200) # starting tpg population size
parser.add_option('-m', '--popMax', type='int', dest='popSizeMax', default=750) # population size to work up to
parser.add_option('-w', '--workers', type='int', dest='workers', default=3) # concurrent workers for parallization
parser.add_option('-t', '--type', type='int', dest='trainType', default=0) # method of training
# trainType is 0 for 'all at once', 1 for 'merge', 2 for 'gradual merge'.

(options, args) = parser.parse_args()
