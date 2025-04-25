from pthcfg import *
cfg = PathConfig()

import json
import os
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


def loadMap(role, mode='eng2chi'):
    assert role in ['Victim', 'Criminal'], "role must be 'Victim' or 'Criminal'"
    assert mode in ['eng2chi', 'chi2eng'], "mode must be 'eng2chi' or 'chi2eng'"
    mapPath = os.path.join(cfg.assets_path, "NameMap.json")
    with open(mapPath, 'r') as f:
        mapAll = json.load(f)
    trainDatasetEng2ChiMap = mapAll["trainDatasetEng2Chi"][role]
    if mode == 'eng2chi':
        return trainDatasetEng2ChiMap
    else:
        return {v: k for k, v in trainDatasetEng2ChiMap.items()}



def graphSelfEvaluate(bnn: BayesianNetwork,
                      aimColumn: str,
                      dataPath: str = os.path.join(cfg.database_path, "bayesian_criminal_filled.csv"),
                      test_size: float = 0.3,
                      random_state: int = 42) -> dict:
    """
    将训练数据拆成两部分，一部分用于训练贝叶斯网络，另一部分用于测试。
    使用非aimColumn数据作为特征预测aimColumn中的数据，并且计算F1、P、R。

    :param bnn: 已定义结构的 BayesianNetwork 对象
    :param aimColumn: 目标列名称，用于预测
    :param dataPath: CSV 数据文件路径
    :param test_size: 测试集比例，默认为 0.3
    :param random_state: 随机种子，保证可重复性
    :return: 包含 precision、recall、f1 的字典
    """

    # 读取列名表
    # column_map = loadMap(role='Criminal', mode='chi2eng')
    # 读取数据
    data = pd.read_csv(dataPath)
    # 取出在column_map中的列
    # data = data[[i for i in data.columns if i in column_map]]
    # data = data.rename(columns=column_map)

    bnnNodeList = list(bnn.nodes)
    # 取出在bnnNodeList中的列
    data = data[[i for i in data.columns if i in bnnNodeList]]

    # aimColumn = column_map[aimColumn]

    # 校验目标列是否存在
    if aimColumn not in data.columns:
        raise ValueError(f"aimColumn '{aimColumn}' 不在数据中")

    # 分离特征和目标
    X = data.drop(columns=[aimColumn])
    # X = data
    y = data[aimColumn]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # 合并特征和目标，用于估计 CPD
    train_data = pd.concat([X_train, y_train], axis=1)

    # 校验train_data中的节点是否与bnn中的节点相同
    if (not set(train_data.columns).issubset(set(bnn.nodes))) or (not set(bnn.nodes).issubset(set(train_data.columns))):
        raise ValueError("训练数据中的节点与贝叶斯网络中的节点不一致")

    # 使用最大似然估计拟合 CPDs
    bnn.fit(train_data, estimator=MaximumLikelihoodEstimator)

    # 进行变量消元推理
    infer = VariableElimination(bnn)

    y_pred = []
    # 对测试集逐行推理预测
    for _, row in X_test.iterrows():
        try:
            evidence = row.to_dict()
            print(evidence, aimColumn)
            q = infer.query(variables=[aimColumn], evidence=evidence, show_progress=False)
            # 获取概率最大的状态索引
            flat_idx = q.values.argmax()
            state = q.state_names[aimColumn][flat_idx]
            y_pred.append(state)
        except Exception as e:
            print(f"Error occurred while processing row: {row}")
            raise e

    # 计算评估指标
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    metrics = {'precision': precision, 'recall': recall, 'f1': f1}
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return metrics
