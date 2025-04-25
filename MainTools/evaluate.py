from pthcfg import *
cfg = PathConfig()

import tqdm
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
    for _, row in tqdm.tqdm(X_test.iterrows()):
        try:
            evidence = row.to_dict()
            # print(evidence, aimColumn)
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


def enhanced_graphSelfEvaluate(bnn: BayesianNetwork,
                              aimColumn: str,
                              dataPath: str = os.path.join(cfg.database_path, "bayesian_criminal_filled.csv"),
                              test_size: float = 0.3,
                              random_state: int = 42) -> dict:
    """
    改进版的graphSelfEvaluate函数，能处理状态不完整的问题
    
    :param bnn: 已定义结构的 BayesianNetwork 对象
    :param aimColumn: 目标列名称，用于预测
    :param dataPath: CSV 数据文件路径
    :param test_size: 测试集比例，默认为 0.3
    :param random_state: 随机种子，保证可重复性
    :return: 包含 precision、recall、f1 的字典
    """
    # 读取数据
    data = pd.read_csv(dataPath)
    
    # 获取模型中的节点列表
    bnnNodeList = list(bnn.nodes)
    
    # 只保留模型中存在的列
    data = data[[i for i in data.columns if i in bnnNodeList]]
    
    # 创建状态空间字典，记录每个节点的所有可能状态
    state_names = {}
    
    # 对每个节点，获取数据中的所有状态
    for node in bnnNodeList:
        if node in data.columns:
            # 获取数据中该节点的所有唯一值
            node_states = sorted(data[node].unique())
            print(f"节点 {node} 在数据中的唯一状态值: {node_states}")
            
            # 记录节点的状态空间
            state_names[node] = node_states
    
    # 校验目标列是否存在
    if aimColumn not in data.columns:
        raise ValueError(f"aimColumn '{aimColumn}' 不在数据中")

    # 分离特征和目标
    X = data.drop(columns=[aimColumn])
    y = data[aimColumn]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # 合并特征和目标，用于估计 CPD
    train_data = pd.concat([X_train, y_train], axis=1)

    # 校验train_data中的节点是否与bnn中的节点相同
    if (not set(train_data.columns).issubset(set(bnn.nodes))) or (not set(bnn.nodes).issubset(set(train_data.columns))):
        raise ValueError("训练数据中的节点与贝叶斯网络中的节点不一致")

    # 标准方式拟合CPDs
    try:
        # 先尝试用普通方式拟合
        bnn.fit(train_data, estimator=MaximumLikelihoodEstimator)
        print("成功使用标准方式拟合CPDs")
    except Exception as e:
        print(f"标准拟合方式失败: {str(e)}")
        print("尝试手动拟合每个节点CPD...")

        # 移除所有现有的CPDs
        if bnn.cpds:
            for cpd in bnn.cpds:
                bnn.remove_cpds(cpd)
        
        # 手动拟合每个节点的CPD
        estimator = MaximumLikelihoodEstimator(bnn, train_data)
        
        for node in bnnNodeList:
            try:
                # 先尝试正常估计CPD
                cpd = estimator.estimate_cpd(node)
                bnn.add_cpds(cpd)
                print(f"成功为节点 {node} 拟合CPD")
            except Exception as e:
                print(f"无法为节点 {node} 正常拟合CPD: {str(e)}")
                print(f"尝试手动创建节点 {node} 的CPD...")
                
                # 获取节点的父节点
                parents = list(bnn.get_parents(node))
                
                # 获取状态数量
                node_states = state_names[node]
                node_card = len(node_states)
                
                # 获取父节点状态数量
                parent_cards = []
                for parent in parents:
                    parent_states = state_names.get(parent, [0, 1])  # 默认假设是二值变量
                    parent_cards.append(len(parent_states))
                
                # 创建CPD值矩阵（均匀分布）
                total_parent_configs = np.prod(parent_cards) if parent_cards else 1
                cpd_values = np.ones((node_card, int(total_parent_configs))) / node_card
                
                # 创建状态名称映射
                cpd_state_names = {node: node_states}
                for i, parent in enumerate(parents):
                    cpd_state_names[parent] = state_names.get(parent, list(range(parent_cards[i])))
                
                # 创建TabularCPD
                try:
                    new_cpd = TabularCPD(
                        variable=node,
                        variable_card=node_card,
                        values=cpd_values,
                        evidence=parents,
                        evidence_card=parent_cards,
                        state_names=cpd_state_names
                    )
                    bnn.add_cpds(new_cpd)
                    print(f"成功为节点 {node} 创建了均匀分布的CPD")
                except Exception as sub_e:
                    print(f"创建节点 {node} 的CPD时发生错误: {str(sub_e)}")
                    raise ValueError(f"无法为节点 {node} 创建有效的CPD")

    # 确认模型是否有效
    try:
        is_valid = bnn.check_model()
        if not is_valid:
            print("模型验证失败！CPD定义可能有问题。")
            # 尝试修复：检查所有节点是否都有CPD
            for node in bnnNodeList:
                if node not in [cpd.variable for cpd in bnn.get_cpds()]:
                    print(f"节点 {node} 缺少CPD，尝试添加...")
                    # 类似上面的代码创建均匀分布CPD
        else:
            print("模型验证通过！")
    except Exception as e:
        print(f"模型验证时出错: {str(e)}")
        print("继续尝试进行推理...")

    # 进行变量消元推理
    try:
        infer = VariableElimination(bnn)
    except Exception as e:
        print(f"创建推理引擎时出错: {str(e)}")
        print("无法进行后续评估，返回默认指标")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    y_pred = []
    failures = 0
    # 对测试集逐行推理预测
    for idx, row in tqdm.tqdm(X_test.iterrows()):
        try:
            evidence = row.to_dict()
            
            # 检查证据中是否有超出模型范围的状态值
            valid_evidence = {}
            for var, val in evidence.items():
                if var in state_names and val in state_names[var]:
                    valid_evidence[var] = val
                else:
                    print(f"警告: 行 {idx} 中变量 {var} 的值 {val} 不在模型的状态空间内，已忽略")
            
            if not valid_evidence:
                raise ValueError("所有证据变量都无效")
                
            q = infer.query(variables=[aimColumn], evidence=valid_evidence, show_progress=False)
            
            # 获取概率最大的状态索引
            flat_idx = q.values.argmax()
            state = q.state_names[aimColumn][flat_idx]
            y_pred.append(state)
        except Exception as e:
            failures += 1
            # 遇到错误时，使用训练集中目标变量的众数来预测
            # if failures < 5:  # 只记录前几个错误
            #     print(f"行 {idx} 推理错误: {str(e)}")
            #     print(f"错误行数据: {row}")
            
            # 使用众数作为预测值
            most_common = y_train.mode()[0]
            y_pred.append(most_common)
            # if failures < 5:
            #     print(f"使用众数 {most_common} 作为预测值")
    
    if failures > 0:
        print(f"总共有 {failures} / {len(X_test)} 行数据无法正常预测，已使用众数替代")

    # 计算评估指标
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    metrics = {'precision': precision, 'recall': recall, 'f1': f1}
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return metrics