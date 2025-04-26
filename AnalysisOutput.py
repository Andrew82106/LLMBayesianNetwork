import time
from MainTools.pthcfg import PathConfig

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


if __name__ == "__main__":
    pklList = os.listdir(cfg.final_output_path)
    expertPKLList = []
    llmPKLList = []
    for pkl in pklList:
        if pkl.endswith('.pkl'):
            if "Expert" in pkl:
                expertPKLList.append(pkl)
            elif "labData" in pkl:
                llmPKLList.append(pkl)
    
    dfList = []
    aimColumn = 'RecoverTime'
    for expertPKL in expertPKLList:
        for llmPKL in llmPKLList:
            result_dict = compareTwoModels(expertPKL, llmPKL, generateExpertResult(expertPKL), generateLLMResult(llmPKL), AimColumn=aimColumn, save=False)
            # 将字典转换为DataFrame后再添加到列表中
            df = pd.DataFrame(result_dict)
            dfList.append(df)
    
    # 现在dfList中只包含DataFrame对象，可以安全地使用pd.concat
    df = pd.concat(dfList)
    # 去重
    df = df.drop_duplicates()
    df.to_csv(os.path.join(cfg.final_output_path, f"finalLabRes_AimColumn_all_{aimColumn}.csv"), index=False)
