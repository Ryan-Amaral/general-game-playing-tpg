"""
Script for testing performance profile of TPG runs.
Created by: Ryan Amaral
Created on: Mar. 23, 2019
"""

from __future__ import division
from tpg.tpg_trainer import TpgTrainer
from tpg.tpg_agent import TpgAgent
from optparse import OptionParser
from tpg_helpers import *
import random
import datetime
import multiprocessing as mp
from timeit import default_timer as timer
