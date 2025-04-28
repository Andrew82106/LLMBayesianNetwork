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
        G1 = ExpertPKL['result']['Graph']['Criminalhuman'][1]
        llmPKL = readOutputPkl(os.path.join(pthcfg.final_output_path, llmRoute))
        G2 = llmPKL['result']['Graph']['Criminalllm'][1]
        F1, P, R = calc_graph_f1(G1.edges(), G2.edges())
        print(F1, P, R)
        if F1 > maxF1:
            maxF1 = F1
            maxExpertRoute = ExpertRoute
            maxLLMRoute = llmRoute


print(maxExpertRoute)
print(maxLLMRoute)
pklData = readOutputPkl(os.path.join(pthcfg.final_output_path, maxExpertRoute))

print(type(pklData['result']['Graph']['Criminalhuman'][1]))

visualize_bayesian_network(pklData['result']['Graph']['Criminalhuman'][1], os.path.join(pthcfg.figure_output_path, "Criminalhuman.png"))

G1 = pklData['result']['Graph']['Criminalhuman'][1]

pklData = readOutputPkl(os.path.join(pthcfg.final_output_path, maxLLMRoute))

print(type(pklData['result']['Graph']['Criminalllm'][1]))

visualize_bayesian_network(pklData['result']['Graph']['Criminalllm'][1], os.path.join(pthcfg.figure_output_path, "Criminalllm.png"))

G2 = pklData['result']['Graph']['Criminalllm'][1]

print(pklData['parameter'])

# print(calc_graph_f1(G1.edges(), G2.edges()))


ExpertBayesianNetworkCriminal = readOutputPkl(os.path.join(pthcfg.final_output_path, "ExpertBayesianNetworkCriminal.pkl"))
visualize_bayesian_network(ExpertBayesianNetworkCriminal['result']['Graph']['Criminalhuman'][1], os.path.join(pthcfg.figure_output_path, "Criminalhuman.png"))
ExpertBayesianNetworkVictim = readOutputPkl(os.path.join(pthcfg.final_output_path, "ExpertBayesianNetworkVictim.pkl"))
visualize_bayesian_network(ExpertBayesianNetworkVictim['result']['Graph']['Victimhuman'][1], os.path.join(pthcfg.figure_output_path, "Victimhuman.png"))
LLMBayesianNetworkCriminal = readOutputPkl(os.path.join(pthcfg.final_output_path, "LLMBayesianNetworkCriminal.pkl"))
visualize_bayesian_network(LLMBayesianNetworkCriminal['result']['Graph']['Criminalllm'][1], os.path.join(pthcfg.figure_output_path, "Criminalllm.png"))
LLMBayesianNetworkVictim = readOutputPkl(os.path.join(pthcfg.final_output_path, "LLMBayesianNetworkVictim.pkl"))
visualize_bayesian_network(LLMBayesianNetworkVictim['result']['Graph']['Victimllm'][1], os.path.join(pthcfg.figure_output_path, "Victimllm.png"))

ExpertCriminalGList = list(ExpertBayesianNetworkCriminal['result']['Graph']['Criminalhuman'][1].edges())
LLMCriminalGList = list(LLMBayesianNetworkCriminal['result']['Graph']['Criminalllm'][1].edges())
print("嫌疑人模型专家与大语言模型对比:" , calc_graph_f1(ExpertCriminalGList, LLMCriminalGList))
ExpertVictimGList = list(ExpertBayesianNetworkVictim['result']['Graph']['Victimhuman'][1].edges())
LLMVictimGList = list(LLMBayesianNetworkVictim['result']['Graph']['Victimllm'][1].edges())
print("受害者模型专家与大语言模型对比:" , calc_graph_f1(ExpertVictimGList, LLMVictimGList))

LLMCriminalGList.append(("SuspectAge", "RecoverTime"))
LLMCriminalGList.append(("SuspectAge", "SuspectGender"))
LLMCriminalGList.append(("SuspectEducation", "CommunicationMethod"))

print("嫌疑人模型专家与大语言模型对比(DS前):" , calc_graph_f1(ExpertCriminalGList, LLMCriminalGList))

LLMVictimGList.append(('VictimSex', 'CityEconomy'))
LLMVictimGList.append(('VictimSex', 'CommunicationMethod'))
LLMVictimGList.append(('SpecialDate', 'VictimOccupation'))
LLMVictimGList.append(('VictimEducation', 'SpecialDate'))
print("受害者模型专家与大语言模型对比(DS前):" , calc_graph_f1(ExpertVictimGList, LLMVictimGList))