from datetime import datetime
import pickle
from MainTools.pthcfg import *
from MainTools.DrawGraph import visualize_bayesian_network

pthcfg = PathConfig()

from MainTools.utils import readOutputPkl

pklData = readOutputPkl(os.path.join(pthcfg.final_output_path, "ExpertResult_labData_time_2025-04-2511_00_51.pkl"))

print(type(pklData['result']['Graph']['Victimhuman'][1]))

visualize_bayesian_network(pklData['result']['Graph']['Victimhuman'][1], os.path.join(pthcfg.figure_output_path, "Victimhuman.png"))
