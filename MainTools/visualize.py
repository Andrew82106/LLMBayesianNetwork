import pandas as pd
from pthcfg import PathConfig
import os
pthcfg = PathConfig()

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def extractDataFromCSV(csv_path, LlMName, DatasetName):
    epsList = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    df_ = pd.read_csv(csv_path)
    df_ = df_[(df_['llmName']==LlMName)&(df_['dataset']==DatasetName)]
    
    # 检查每个eps下是否有数据
    data_counts = {eps: len(df_[df_['eps'] == eps]) for eps in epsList}
    print(f"数据统计: {data_counts}")
    
    # 找出所有eps值中样本数量最小的值
    min_count = min(data_counts.values()) if data_counts.values() else 0
    if min_count == 0:
        print(f"警告: 某些eps值没有数据。请检查数据集中是否包含LLM名称：{LlMName}和数据集：{DatasetName}")
    
    # 构建确保长度一致的字典
    epsResultF1 = {}
    epsResultP = {}
    epsResultR = {}
    
    for eps in epsList:
        eps_data = df_[df_['eps'] == eps]
        if len(eps_data) > 0:
            epsResultF1[eps] = list(eps_data['f1'])
            epsResultP[eps] = list(eps_data['precision'])
            epsResultR[eps] = list(eps_data['recall'])
            
    # 检查是否有数据
    if not epsResultF1:
        print(f"错误: 没有找到任何数据！请检查LLM名称：{LlMName}和数据集：{DatasetName}是否正确。")
        # 返回空DataFrame
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
    # 检查长度一致性
    lengths = [len(values) for values in epsResultF1.values()]
    if len(set(lengths)) > 1:
        print(f"警告: 不同eps值下的数据点数量不一致: {lengths}")
        print("将所有列表裁剪到相同长度以确保DataFrame可以创建")
        
        # 找到最小长度
        min_length = min(lengths)
        
        # 裁剪所有列表到最小长度
        for eps in epsResultF1.keys():
            epsResultF1[eps] = epsResultF1[eps][:min_length]
            epsResultP[eps] = epsResultP[eps][:min_length]
            epsResultR[eps] = epsResultR[eps][:min_length]
    
    # 创建DataFrame
    epsResultF1DF = pd.DataFrame(epsResultF1)
    epsResultPDF = pd.DataFrame(epsResultP)
    epsResultRDF = pd.DataFrame(epsResultR)
    
    return epsResultF1DF, epsResultPDF, epsResultRDF


def extractDataFromCSV_NoneBayesian(csv_path, ModelName, DatasetName):
    epsList = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    df_ = pd.read_csv(csv_path)
    df_ = df_[(df_['model']==ModelName)&(df_['dataset']==DatasetName)]
    
    # 检查每个eps下是否有数据
    data_counts = {eps: len(df_[df_['eps'] == eps]) for eps in epsList}
    print(f"数据统计(非贝叶斯网络): {data_counts}")
    
    # 找出所有eps值中样本数量最小的值
    min_count = min(data_counts.values()) if data_counts.values() else 0
    if min_count == 0:
        print(f"警告: 某些eps值没有数据。请检查数据集中是否包含模型名称：{ModelName}和数据集：{DatasetName}")
    
    # 构建确保长度一致的字典
    epsResultF1 = {}
    epsResultP = {}
    epsResultR = {}
    
    for eps in epsList:
        eps_data = df_[df_['eps'] == eps]
        if len(eps_data) > 0:
            epsResultF1[eps] = list(eps_data['f1'])
            epsResultP[eps] = list(eps_data['precision'])
            epsResultR[eps] = list(eps_data['recall'])
    
    # 检查是否有数据
    if not epsResultF1:
        print(f"错误: 没有找到任何数据！请检查模型名称：{ModelName}和数据集：{DatasetName}是否正确。")
        # 返回空DataFrame
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
    # 检查长度一致性
    lengths = [len(values) for values in epsResultF1.values()]
    if len(set(lengths)) > 1:
        print(f"警告: 不同eps值下的数据点数量不一致: {lengths}")
        print("将所有列表裁剪到相同长度以确保DataFrame可以创建")
        
        # 找到最小长度
        min_length = min(lengths)
        
        # 裁剪所有列表到最小长度
        for eps in epsResultF1.keys():
            epsResultF1[eps] = epsResultF1[eps][:min_length]
            epsResultP[eps] = epsResultP[eps][:min_length]
            epsResultR[eps] = epsResultR[eps][:min_length]
    
    # 创建DataFrame
    epsResultF1DF = pd.DataFrame(epsResultF1)
    epsResultPDF = pd.DataFrame(epsResultP)
    epsResultRDF = pd.DataFrame(epsResultR)
    
    return epsResultF1DF, epsResultPDF, epsResultRDF


def drawBoxPlot(epsResultF1DF, epsResultPDF, epsResultRDF, LlMName, DatasetName, AimColumn, savePath=None):
    # 检查是否有数据
    if epsResultF1DF.empty or epsResultPDF.empty or epsResultRDF.empty:
        print(f"警告: 没有足够的数据来绘制箱线图。LLM名称: {LlMName}, 数据集: {DatasetName}")
        # 创建一个简单的错误信息图
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"无法绘制箱线图: 未找到 {LlMName} 模型在 {DatasetName} 数据集上的数据",
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        if savePath:
            plt.savefig(savePath)
        else:
            plt.show()
        return
    
    # 创建三个指标的并排箱线图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # F1箱线图  
    sns.boxplot(data=epsResultF1DF, ax=axes[0])
    axes[0].set_title(f'{LlMName}模型{DatasetName}数据集{AimColumn}指标F1值箱线图')
    axes[0].set_xlabel('测试集占比')
    axes[0].set_ylabel('F1值')
    axes[0].grid(True)

    # Precision箱线图
    sns.boxplot(data=epsResultPDF, ax=axes[1])
    axes[1].set_title(f'{LlMName}模型{DatasetName}数据集{AimColumn}指标Precision值箱线图')
    axes[1].set_xlabel('测试集占比')
    axes[1].set_ylabel('Precision值')
    axes[1].grid(True)

    # Recall箱线图
    sns.boxplot(data=epsResultRDF, ax=axes[2])
    axes[2].set_title(f'{LlMName}模型{DatasetName}数据集{AimColumn}指标Recall值箱线图')
    axes[2].set_xlabel('测试集占比')
    axes[2].set_ylabel('Recall值')
    axes[2].grid(True)

    plt.tight_layout()
    if savePath:
        plt.savefig(savePath)
    else:
        plt.show()



