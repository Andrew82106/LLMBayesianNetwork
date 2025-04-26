import os
from MainTools.pthcfg import PathConfig
pthcfg = PathConfig()
from MainTools.visualize import extractDataFromCSV, extractDataFromCSV_NoneBayesian, drawBoxPlot


def visualizeData(csvPath, aimColumn, LLMName, DatasetName):
    print(f"\n处理数据: LLM={LLMName}, 数据集={DatasetName}, 指标={aimColumn}")
    epsResultF1DF, epsResultPDF, epsResultRDF = extractDataFromCSV(csvPath, LLMName, DatasetName)
    drawBoxPlot(epsResultF1DF, epsResultPDF, epsResultRDF, LLMName, DatasetName, aimColumn, savePath=os.path.join(pthcfg.figure_output_path, f'{LLMName}_{DatasetName}_{aimColumn}_boxplot.png'))


if __name__ == '__main__':
    # loss 指标的可视化
    csvPath = os.path.join(pthcfg.final_output_path, 'finalLabRes_AimColumn_all_loss.csv')
    
    # 为不同的LLM和数据集生成图表
    visualizeData(csvPath, "loss", "Qwen", "Criminal")
    visualizeData(csvPath, "loss", "QWQ", "Criminal")
    visualizeData(csvPath, "loss", "QWQ_plus", "Criminal")  
    visualizeData(csvPath, "loss", "ChatGLM4Flash", "Criminal")
    visualizeData(csvPath, "loss", "Qwen", "Victim")
    visualizeData(csvPath, "loss", "QWQ", "Victim")
    visualizeData(csvPath, "loss", "QWQ_plus", "Victim")
    visualizeData(csvPath, "loss", "ChatGLM4Flash", "Victim")
    
    # RecoverTime 指标的可视化
    csvPath = os.path.join(pthcfg.final_output_path, 'finalLabRes_AimColumn_all_RecoverTime.csv')
    visualizeData(csvPath, "RecoverTime", "Qwen", "Criminal")
    visualizeData(csvPath, "RecoverTime", "QWQ", "Criminal")
    visualizeData(csvPath, "RecoverTime", "QWQ_plus", "Criminal")
    visualizeData(csvPath, "RecoverTime", "ChatGLM4Flash", "Criminal")
    visualizeData(csvPath, "RecoverTime", "Qwen", "Victim")
    visualizeData(csvPath, "RecoverTime", "QWQ", "Victim")
    visualizeData(csvPath, "RecoverTime", "QWQ_plus", "Victim")
    visualizeData(csvPath, "RecoverTime", "ChatGLM4Flash", "Victim")


