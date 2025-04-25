import time
import pandas as pd
import numpy as np
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import tqdm
from MainTools.pthcfg import PathConfig
from MainTools.otherMethodsEvaluate import randomForestPredict, decisionTreePredict, knnPredict, svmPredict

cfg = PathConfig()

import os.path
from MainTools.utils import *
import pprint
from MainTools.DrawGraph import *
from MainTools.evaluate import *


def generateLLMResult(filename):
    result = readOutputPkl(os.path.join(cfg.final_output_path, filename))
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
    return result


def generateExpertResult(filename):
    result = readOutputPkl(os.path.join(cfg.final_output_path, filename))
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


def evaluateF1(aimGraph: BayesianNetwork, dataPath: str, epsList=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], aimColumn='Loss'):
    res_list = []
    for eps in epsList:
        evaluateResult = enhanced_graphSelfEvaluate(
            aimGraph,
            aimColumn, 
            dataPath,
            test_size=eps,
            random_state=(time.time_ns()) % 4294967295
        )
        res_list.append({"result": evaluateResult, "eps": eps})
    return res_list


def evaluateRandomForest(dataPath: str, epsList=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], aimColumn='Loss'):
    res_list = []
    for eps in epsList:
        res = randomForestPredict(dataPath, aimColumn, test_size=eps)
        res_list.append({"result": res, "eps": eps})
    return res_list

def evaluateDecisionTree(dataPath: str, epsList=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], aimColumn='Loss'):
    res_list = []
    for eps in epsList:
        res = decisionTreePredict(dataPath, aimColumn, test_size=eps)
        res_list.append({"result": res, "eps": eps})
    return res_list

def evaluateKNN(dataPath: str, epsList=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], aimColumn='Loss'):
    res_list = []
    for eps in epsList:
        res = knnPredict(dataPath, aimColumn, test_size=eps)
        res_list.append({"result": res, "eps": eps})
    return res_list

def evaluateSVM(dataPath: str, epsList=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], aimColumn='Loss'):
    res_list = []
    for eps in epsList:
        res = svmPredict(dataPath, aimColumn, test_size=eps)
        res_list.append({"result": res, "eps": eps})
    return res_list


if __name__ == "__main__":
    finalLabRes = {
        "eps": [],
        "dataset": [],
        "expertType": [],
        "model": [],
        "f1": [],
        "precision": [],
        "recall": [],
    }
    AimColumn = 'RecoverTime'
    expertResult = generateExpertResult('ExpertResult_labData_time=2025-04-2509_15_59.pkl')
    llmResult = generateLLMResult('labData_time=2025-04-22_18-04-22.pkl')

    # Criminal result evaluation
    expGraph: BayesianNetwork = expertResult['result']['Graph']['Criminalhuman'][1]
    expGraph.remove_node("CallPolice")

    llmGraph: BayesianNetwork = llmResult['result']['Graph']['Criminalllm'][1]
    llmGraph.remove_node("CallPolice")

    expF1resCriminal = evaluateF1(expGraph, os.path.join(cfg.database_path, "bayesian_criminal_filled.csv"), aimColumn=AimColumn)
    llmF1resCriminal = evaluateF1(llmGraph, os.path.join(cfg.database_path, "bayesian_criminal_filled.csv"), aimColumn=AimColumn)

    # Victim result evaluation
    expGraph: BayesianNetwork = expertResult['result']['Graph']['Victimhuman'][1]
    expGraph.remove_node("CallPolice")

    llmGraph: BayesianNetwork = llmResult['result']['Graph']['Victimllm'][1]
    llmGraph.remove_node("CallPolice")

    expF1resVictim = evaluateF1(expGraph, os.path.join(cfg.database_path, "bayesian_victim_and_others_filled.csv"), aimColumn=AimColumn)
    llmF1resVictim = evaluateF1(llmGraph, os.path.join(cfg.database_path, "bayesian_victim_and_others_filled.csv"), aimColumn=AimColumn)

    # 随机森林预测
    RFresCriminal = evaluateRandomForest(os.path.join(cfg.database_path, "bayesian_criminal_filled.csv"), aimColumn=AimColumn)
    RFresVictim = evaluateRandomForest(os.path.join(cfg.database_path, "bayesian_victim_and_others_filled.csv"), aimColumn=AimColumn)

    # 决策树预测
    DTresCriminal = evaluateDecisionTree(os.path.join(cfg.database_path, "bayesian_criminal_filled.csv"), aimColumn=AimColumn)
    DTresVictim = evaluateDecisionTree(os.path.join(cfg.database_path, "bayesian_victim_and_others_filled.csv"), aimColumn=AimColumn)

    # KNN预测
    KNNresCriminal = evaluateKNN(os.path.join(cfg.database_path, "bayesian_criminal_filled.csv"), aimColumn=AimColumn)
    KNNresVictim = evaluateKNN(os.path.join(cfg.database_path, "bayesian_victim_and_others_filled.csv"), aimColumn=AimColumn)

    # SVM预测
    SVMresCriminal = evaluateSVM(os.path.join(cfg.database_path, "bayesian_criminal_filled.csv"), aimColumn=AimColumn)
    SVMresVictim = evaluateSVM(os.path.join(cfg.database_path, "bayesian_victim_and_others_filled.csv"), aimColumn=AimColumn)

    
    print("Criminal result evaluation:")
    for res in expF1resCriminal:
        finalLabRes["eps"].append(res["eps"])
        finalLabRes["dataset"].append("Criminal")
        finalLabRes["model"].append("Bayesian Network")
        finalLabRes["expertType"].append("human")
        finalLabRes["f1"].append(res["result"]["f1"])
        finalLabRes["precision"].append(res["result"]["precision"])
        finalLabRes["recall"].append(res["result"]["recall"])
    for res in llmF1resCriminal:
        finalLabRes["eps"].append(res["eps"])
        finalLabRes["dataset"].append("Criminal")
        finalLabRes["model"].append("Bayesian Network")
        finalLabRes["expertType"].append("llm")
        finalLabRes["f1"].append(res["result"]["f1"])
        finalLabRes["precision"].append(res["result"]["precision"])
        finalLabRes["recall"].append(res["result"]["recall"])
    for res in RFresCriminal:
        finalLabRes["eps"].append(res["eps"])
        finalLabRes["dataset"].append("Criminal")
        finalLabRes["model"].append("Random Forest")
        finalLabRes["expertType"].append("Non-expert")
        finalLabRes["f1"].append(res["result"]["f1"])
        finalLabRes["precision"].append(res["result"]["precision"])
        finalLabRes["recall"].append(res["result"]["recall"])
    for res in DTresCriminal:
        finalLabRes["eps"].append(res["eps"])
        finalLabRes["dataset"].append("Criminal")
        finalLabRes["model"].append("Decision Tree")
        finalLabRes["expertType"].append("Non-expert")
        finalLabRes["f1"].append(res["result"]["f1"])
        finalLabRes["precision"].append(res["result"]["precision"])
        finalLabRes["recall"].append(res["result"]["recall"])
    for res in KNNresCriminal:
        finalLabRes["eps"].append(res["eps"])
        finalLabRes["dataset"].append("Criminal")
        finalLabRes["model"].append("KNN")
        finalLabRes["expertType"].append("Non-expert")
        finalLabRes["f1"].append(res["result"]["f1"])
        finalLabRes["precision"].append(res["result"]["precision"])
        finalLabRes["recall"].append(res["result"]["recall"])
    for res in SVMresCriminal:
        finalLabRes["eps"].append(res["eps"])
        finalLabRes["dataset"].append("Criminal")
        finalLabRes["model"].append("SVM")
        finalLabRes["expertType"].append("Non-expert")
        finalLabRes["f1"].append(res["result"]["f1"])
        finalLabRes["precision"].append(res["result"]["precision"])
        finalLabRes["recall"].append(res["result"]["recall"])        
        
    # pprint.pprint(expF1resCriminal)
    # pprint.pprint(llmF1resCriminal)
    # pprint.pprint(RFresCriminal)

    print("Victim result evaluation:")
    for res in expF1resVictim:
        finalLabRes["eps"].append(res["eps"])
        finalLabRes["dataset"].append("Victim")
        finalLabRes["model"].append("Bayesian Network")
        finalLabRes["expertType"].append("human")
        finalLabRes["f1"].append(res["result"]["f1"])
        finalLabRes["precision"].append(res["result"]["precision"])
        finalLabRes["recall"].append(res["result"]["recall"])
    for res in llmF1resVictim:
        finalLabRes["eps"].append(res["eps"])
        finalLabRes["dataset"].append("Victim")
        finalLabRes["model"].append("Bayesian Network")
        finalLabRes["expertType"].append("llm")
        finalLabRes["f1"].append(res["result"]["f1"])
        finalLabRes["precision"].append(res["result"]["precision"])
        finalLabRes["recall"].append(res["result"]["recall"])
    for res in RFresVictim:
        finalLabRes["eps"].append(res["eps"])
        finalLabRes["dataset"].append("Victim")
        finalLabRes["model"].append("Random Forest")
        finalLabRes["expertType"].append("Non-expert")
        finalLabRes["f1"].append(res["result"]["f1"])
        finalLabRes["precision"].append(res["result"]["precision"])
        finalLabRes["recall"].append(res["result"]["recall"])
    for res in DTresVictim:
        finalLabRes["eps"].append(res["eps"])
        finalLabRes["dataset"].append("Victim")
        finalLabRes["model"].append("Decision Tree")
        finalLabRes["expertType"].append("Non-expert")
        finalLabRes["f1"].append(res["result"]["f1"])
        finalLabRes["precision"].append(res["result"]["precision"])
        finalLabRes["recall"].append(res["result"]["recall"])
    for res in KNNresVictim:
        finalLabRes["eps"].append(res["eps"])
        finalLabRes["dataset"].append("Victim")
        finalLabRes["model"].append("KNN")
        finalLabRes["expertType"].append("Non-expert")
        finalLabRes["f1"].append(res["result"]["f1"])
        finalLabRes["precision"].append(res["result"]["precision"])
        finalLabRes["recall"].append(res["result"]["recall"])
    for res in SVMresVictim:
        finalLabRes["eps"].append(res["eps"])
        finalLabRes["dataset"].append("Victim")
        finalLabRes["model"].append("SVM")
        finalLabRes["expertType"].append("Non-expert")
        finalLabRes["f1"].append(res["result"]["f1"])
        finalLabRes["precision"].append(res["result"]["precision"])
        finalLabRes["recall"].append(res["result"]["recall"])
        
    # pprint.pprint(expF1resVictim)
    # pprint.pprint(llmF1resVictim)
    # pprint.pprint(RFresVictim)

    pd.DataFrame(finalLabRes).to_csv(os.path.join(cfg.final_output_path, f"finalLabRes_AimColumn_{AimColumn}.csv"), index=False)