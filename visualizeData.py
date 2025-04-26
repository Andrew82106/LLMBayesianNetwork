import os
from MainTools.pthcfg import PathConfig
pthcfg = PathConfig()
from MainTools.visualize import extractDataFromCSV, extractDataFromCSV_NoneBayesian, drawBoxPlot


def visualizeData(csvPath, aimColumn, LLMName, DatasetName):
    print(f"\n处理数据: LLM={LLMName}, 数据集={DatasetName}, 指标={aimColumn}")
    epsResultF1DF, epsResultPDF, epsResultRDF = extractDataFromCSV(csvPath, LLMName, DatasetName)
    drawBoxPlot(epsResultF1DF, epsResultPDF, epsResultRDF, LLMName, DatasetName, aimColumn, savePath=os.path.join(pthcfg.figure_output_path, f'{LLMName}_{DatasetName}_{aimColumn}_boxplot.png'))


if __name__ == '__main__':
    # 为不同的LLM和数据集生成图表
    DatasetName = ["Criminal", "Victim"]
    LLMName = ["Qwen", "QWQ", "QWQ_plus", "ChatGLM4Flash"]

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


