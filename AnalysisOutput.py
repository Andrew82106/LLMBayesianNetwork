from MainTools.pthcfg import PathConfig

cfg = PathConfig()

import os.path
from MainTools.utils import *
import pprint
from MainTools.DrawGraph import *
from MainTools.evaluate import *


def generateLLMResult(filename):
    result = readOutputPkl(filename)
    # print(result['result']['Graph']['Criminalhuman'][1].edges())
    # print(result['result']['Graph']['Victimhuman'][1].edges())
    pprint.pprint(result['parameter'])
    visualize_multiple_bayesian_networks(
        [result['result']['Graph']['Victimllm'][0],
         result['result']['Graph']['Victimllm'][1],
         result['result']['Graph']['Criminalllm'][0],
         result['result']['Graph']['Criminalllm'][1]],
        os.path.join(cfg.log_pth, 'LLM_multiple.png'),
        ['Victimllm', 'Victimllm_DS', 'Criminalllm', 'Criminalllm_DS'],
        figsize=(24, 24)
    )


def generateExpertResult(filename):
    result = readOutputPkl(filename)
    pprint.pprint(result['parameter'])
    visualize_multiple_bayesian_networks(
        [result['result']['Graph']['Victimhuman'][0],
         result['result']['Graph']['Victimhuman'][1],
         result['result']['Graph']['Criminalhuman'][0],
         result['result']['Graph']['Criminalhuman'][1]],
        os.path.join(cfg.log_pth, 'Expert_multiple.png'),
        ['Victimhuman', 'Victimhuman_DS', 'Criminalhuman', 'Criminalhuman_DS'],
    )
    return result


def evaluateGraph(bnn, aimColumn, dataPath, test_size, random_state):
    return graphSelfEvaluate(bnn, aimColumn, dataPath, test_size, random_state)



expertResult = generateExpertResult('ExpertResult_labData_time=2025-04-2509_15_59.pkl')

aimGraph: BayesianNetwork = expertResult['result']['Graph']['Criminalhuman'][1]
aimGraph.remove_node("CallPolice")
evaluateResult = evaluateGraph(
    aimGraph,
    '金额损失_code',
    os.path.join(cfg.database_path, "bayesian_criminal_filled.csv"),
    test_size=0.3,
    random_state=42
)
print(evaluateResult)
