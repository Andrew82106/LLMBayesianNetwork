import os
from MainTools.pthcfg import PathConfig
pthcfg = PathConfig()
from MainTools.visualize import extractDataFromCSV, extractDataFromCSV_NoneBayesian, drawBoxPlot
import numpy as np

def visualizeData(csvPath, aimColumn, LLMName, DatasetName):
    print(f"\n处理数据: LLM={LLMName}, 数据集={DatasetName}, 指标={aimColumn}")
    epsResultF1DF, epsResultPDF, epsResultRDF = extractDataFromCSV(csvPath, LLMName, DatasetName)
    drawBoxPlot(epsResultF1DF, epsResultPDF, epsResultRDF, LLMName, DatasetName, aimColumn, savePath=os.path.join(pthcfg.figure_output_path, f'{LLMName}_{DatasetName}_{aimColumn}_boxplot.png'))


def summaryData(csvPath, aimColumn, LLMName, DatasetName):
    print(f"\n处理数据: LLM={LLMName}, 数据集={DatasetName}, 指标={aimColumn}")
    epsResultF1DF, epsResultPDF, epsResultRDF = extractDataFromCSV(csvPath, LLMName, DatasetName)
    columns = list(epsResultF1DF.columns)
    dataF1 = {}
    dataP = {}
    dataR = {}

    for eps in columns:
        dataF1[eps] = list(epsResultF1DF[eps])
        dataP[eps] = list(epsResultPDF[eps])
        dataR[eps] = list(epsResultRDF[eps])
    # 计算F1、P、R的平均值
    dataF1_mean = {}
    dataP_mean = {}
    dataR_mean = {}
    for eps in columns:
        dataF1_mean[eps] = np.mean(dataF1[eps])
        dataP_mean[eps] = np.mean(dataP[eps])
        dataR_mean[eps] = np.mean(dataR[eps])

    # 计算F1、P、R的方差
    dataF1_var = {}
    dataP_var = {}
    dataR_var = {}
    for eps in columns:
        dataF1_var[eps] = np.var(dataF1[eps])
        dataP_var[eps] = np.var(dataP[eps])
        dataR_var[eps] = np.var(dataR[eps])

    # 计算F1、P、R的最大值
    dataF1_max = {}
    dataP_max = {}
    dataR_max = {}
    for eps in columns:
        dataF1_max[eps] = np.max(dataF1[eps])
        dataP_max[eps] = np.max(dataP[eps])
        dataR_max[eps] = np.max(dataR[eps])
    print(f"数据集{DatasetName}, 目标指标{aimColumn}，LLM{LLMName}的计算结果：")
    print(f"F1的平均值：{dataF1_mean}")
    print(f"P的平均值：{dataP_mean}")
    print(f"R的平均值：{dataR_mean}")
    print(f"F1的方差：{dataF1_var}")
    print(f"P的方差：{dataP_var}")
    print(f"R的方差：{dataR_var}")
    print(f"F1的最大值：{dataF1_max}")
    print(f"P的最大值：{dataP_max}")
    print(f"R的最大值：{dataR_max}")
    # 打印平均值最高
    return dataF1_mean, dataP_mean, dataR_mean, dataF1_var, dataP_var, dataR_var


if __name__ == '__main__':
    # 为不同的LLM和数据集生成图表
    DatasetName = ["Criminal", "Victim"]
    LLMName = ["Qwen", "QWQ", "QWQ_plus", "ChatGLM4Flash"]

    # loss 指标的可视化
    csvPath = os.path.join(pthcfg.final_output_path, 'finalLabRes_AimColumn_all_loss.csv')
    for dataset in DatasetName:
        for llm in LLMName:
            summaryData(csvPath, "loss", llm, dataset)

    # loss 指标的可视化
    csvPath = os.path.join(pthcfg.final_output_path, 'finalLabRes_AimColumn_all_RecoverTime.csv')
    for dataset in DatasetName:
        for llm in LLMName:
            summaryData(csvPath, "loss", llm, dataset)
    """
    # loss 指标的可视化
    csvPath = os.path.join(pthcfg.final_output_path, 'finalLabRes_AimColumn_all_loss.csv')
    for dataset in DatasetName:
        for llm in LLMName:
            visualizeData(csvPath, "loss", llm, dataset)

    
    # RecoverTime 指标的可视化
    csvPath = os.path.join(pthcfg.final_output_path, 'finalLabRes_AimColumn_all_RecoverTime.csv')
    for dataset in DatasetName:
        for llm in LLMName:
            visualizeData(csvPath, "RecoverTime", llm, dataset)

    """
