from datetime import datetime
import pickle
from MainTools.pthcfg import *
pthcfg = PathConfig()
from MainTools.DrawGraph import visualize_bayesian_network
from MainTools.utils import calc_graph_f1
from MainTools.utils import readOutputPkl

pklRouteList = os.listdir(pthcfg.final_output_path)
ExpertList = []
llmList = []

for pklRoute in pklRouteList:
    if "Expert" in pklRoute:
        ExpertList.append(pklRoute)
    elif "labData" in pklRoute:
        llmList.append(pklRoute)


print(ExpertList)
print(llmList)
maxF1 = 0
maxExpertRoute = None
maxLLMRoute = None

for ExpertRoute in ExpertList:
    for llmRoute in llmList:
        ExpertPKL = readOutputPkl(os.path.join(pthcfg.final_output_path, ExpertRoute))
        G1 = ExpertPKL['result']['Graph']['Victimhuman'][1]
        llmPKL = readOutputPkl(os.path.join(pthcfg.final_output_path, llmRoute))
        G2 = llmPKL['result']['Graph']['Victimllm'][1]
        F1, P, R = calc_graph_f1(G1.edges(), G2.edges())
        if F1 > maxF1:
            maxF1 = F1
            maxExpertRoute = ExpertRoute
            maxLLMRoute = llmRoute


print(maxExpertRoute)
print(maxLLMRoute)

pklData = readOutputPkl(os.path.join(pthcfg.final_output_path, maxExpertRoute))

print(type(pklData['result']['Graph']['Victimhuman'][1]))

visualize_bayesian_network(pklData['result']['Graph']['Victimhuman'][1], os.path.join(pthcfg.figure_output_path, "Victimhuman.png"))

G1 = pklData['result']['Graph']['Victimhuman'][1]

pklData = readOutputPkl(os.path.join(pthcfg.final_output_path, maxLLMRoute))

print(type(pklData['result']['Graph']['Victimllm'][1]))

visualize_bayesian_network(pklData['result']['Graph']['Victimllm'][1], os.path.join(pthcfg.figure_output_path, "Victimllm.png"))

G2 = pklData['result']['Graph']['Victimllm'][1]

print(pklData['parameter'])

print(calc_graph_f1(G1.edges(), G2.edges()))
