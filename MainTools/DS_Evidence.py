import json

from pthcfg import *
import time

pthcfg = PathConfig()

import numpy as np
from Expert import *


def dempster_rule(m1, m2):
    """
    使用Dempster合成规则合并两个证据源的BPA。

    参数:
    m1, m2 -- 分别为两个专家对某条边的新增和删除的置信度

    返回:
    m_result -- 合并后的BPA字典
    """
    m_result = {}
    normalization_factor = 1  # 归一化因子

    # 计算合成后的BPA
    for A in m1:
        for B in m2:
            intersection = A & B
            if intersection:
                m_result[intersection] = m_result.get(intersection, 0) + m1[A] * m2[B]

    # 计算冲突部分
    conflict = 0
    for A in m1:
        for B in m2:
            intersection = A & B
            if not intersection:  # 无交集即为冲突部分
                conflict += m1[A] * m2[B]

    if conflict == 0:
        normalization_factor = 1
    else:
        normalization_factor = 1 / (1 - conflict)

    # 归一化结果
    for A in m_result:
        m_result[A] *= normalization_factor

    return m_result


def calculate_edge_confidence(edges_, expert_opinions_, options_):
    """
    计算每条边新增和删除的融合置信度。

    参数:
    edges_ -- 边的信息列表，例如 [('A', 'B'), ('B', 'C')]，表示边(A->B)和(B->C)
    expert_opinions_ -- 一个列表，包含每条边的多个专家的意见，
                        每个元素是一个字典，表示该条边的新增和删除置信度。

    返回:
    result -- 每条边的融合置信度字典，格式如下：
              {('A', 'B'): {'新增边': 0.75, '删除边': 0.25},
               ('B', 'C'): {'新增边': 0.65, '删除边': 0.35}}
    """
    result = {}

    # 遍历每条边
    for i, edge__ in enumerate(edges_):
        # edge_name = f"边{i + 1}"
        edge_name = f"{edge__[0]}->{edge__[1]}"
        m_combined = {options_[j]: 0 for j in range(len(options_))}

        # 对每个专家的意见进行合并
        for opinion in expert_opinions_:
            if edge_name not in opinion:
                continue
            opinion_edge = opinion.get(edge_name, {})

            # 每个专家对每条边给出新增和删除的置信度
            m1 = {frozenset([options_[j]]): opinion_edge.get(options_[j], 0) for j in range(len(options_))}
            # 合并证据
            m_result = dempster_rule(m1, m1)

            # 合并结果存入字典
            for j in range(len(options_)):
                m_combined[options_[j]] += m_result.get(frozenset([options_[j]]), 0)
        if m_combined['删除边'] == 0 and m_combined['保留边'] == 0:
            continue
        # 归一化结果
        total_confidence = sum([m_combined[options_[j]] for j in range(len(options_))])
        if total_confidence != 0:

            for j in range(len(options_)):
                m_combined[options_[j]] /= total_confidence

        result[edge__] = m_combined

    return result


def D_S_Experts(
        nodes_name_lst_: list,
        nodes_info_list: list,
        Graph: list,
        options_: list,
        debug=False,
        fresh=True,
        expert_list=None,
        aim=None
):
    """
    使用D-S证据理论融合多专家意见修正贝叶斯网络拓扑结构


    :param aim:
    :param nodes_name_lst_: 节点名称列表，用于索引节点信息
            eg: ['BirthAsphyxia', 'HypDistrib', 'HypoxiaInO2']
    :param nodes_info_list: 与节点名称对应的语义解释列表
            eg: ['BirthAsphyxia的含义', 'HypDistrib的含义', 'HypoxiaInO2的含义']
    :param Graph: 待修正的贝叶斯网络边集合，每条边用元组表示节点关系
            eg: [('BirthAsphyxia', 'HypDistrib'),('BirthAsphyxia', 'HypoxiaInO2')]
    :param options_: 边修正的二元选项集合，第一项为否定操作，第二项为肯定操作
            eg: ['删除边', '保留边']
    :param debug: 调试模式开关，开启时输出详细运行日志（默认False）
    :param numofExpert: 参与决策的专家数量（默认5）
    :param fresh: 缓存刷新开关，为True时重置专家意见缓存（默认True）
    :param expert_list: 预定义专家对象列表，为None时自动初始化默认专家（默认None）
    :return: 经过多专家意见融合后的各边置信度字典

    处理流程：
    1. 初始化缓存文件路径
    2. 根据fresh参数决定是否重置或加载已有专家意见缓存
    3. 自动初始化专家实例（当未提供预定义专家时）
    4. 遍历网络中的每条边，收集各专家对当前边的修正意见
    5. 将实时生成的专家意见持久化到缓存文件
    6. 最终使用D-S理论融合所有专家意见，计算边置信度
    """

    funsionType = str(aim)
    cache_pth = os.path.join(pthcfg.cache_path, "DS_Experts_cache.json")
    expert_opinions_all = {
        "Victim": [{} for _ in range(len(expert_list))],
        "Criminal": [{} for _ in range(len(expert_list))],
    }
    expert_opinions_ = [{} for _ in range(len(expert_list))]
    # 处理缓存文件：若需刷新则清空，否则加载已有数据
    if fresh:
        os.remove(cache_pth)
        # 将expert_opinions_all 写入本地文件
        with open(cache_pth, 'w') as f:
            json.dump(expert_opinions_all, f)

    if os.path.exists(cache_pth):
        # 加载已有缓存数据
        with open(cache_pth, 'r') as f:
            expert_opinions_all = json.load(f)
            if funsionType in expert_opinions_all:
                print(
                    f"in DS_Evidence method: 目标{str(aim)}的意见已有本地缓存，因此直接读取，不经过DS算法")
                expert_opinions_ = expert_opinions_all[funsionType]
            print(f"in DS_Evidence 获取到本地DS算法专家意见缓存，故可以采用缓存数据，下面是缓存数据概要：")
            for index, expert_detail_opinions in enumerate(expert_opinions_):
                print(f"in DS_Evidence, 当前专家{index} 给出各节点间关系共 {len(expert_detail_opinions)} 条")
            print(f"in DS_Evidence 本图中总共 {len(Graph)} 条边")
            print(f"in DS_Evidence 缓存数据读取结束，开始调用专家")


    # 遍历网络中的每条边收集专家意见
    for index, edge_ in enumerate(Graph):
        if debug:
            print("in DS_Evidence 当前处理 ", edge_, f"第{index} 条边，图中共 {len(Graph)} 条边")

        # 解析边信息
        node_from = edge_[0]
        node_to = edge_[1]
        node_from_msg = nodes_info_list[nodes_name_lst_.index(node_from)]
        node_to_msg = nodes_info_list[nodes_name_lst_.index(node_to)]
        nodes_info_dict = {'node_from': node_from_msg, 'node_to': node_to_msg}
        edge_name = f"{node_from}->{node_to}"

        # 获取每个专家对当前边的判断
        for index_, expert in enumerate(expert_list):
            if edge_name in expert_opinions_[index_].keys() and debug:
                print(f"in DS_Evidence 边 {edge_name} 已经在缓存中，存在之前的专家进行了回答，跳过本次处理")
                continue

            # 生成当前专家对边的可信度评估
            relief = generate_relief(
                nodes_pair=[node_from, node_to],
                nodes_info_dict={node_from: nodes_info_dict['node_from'], node_to: nodes_info_dict['node_to']},
                llm_=expert
            )

            # 记录专家意见并实时更新缓存
            expert_opinions_[index_][edge_name] = {options_[0]: 1 - relief, options_[1]: relief}
            with open(cache_pth, 'w') as f:
                # save expert_opinions_
                expert_opinions_all[funsionType] = expert_opinions_
                json.dump(expert_opinions_all, f)
                print(f"in DS_Evidence: 边 {edge_name} 的可信度评估结果已保存到本地缓存")
                if debug:
                    for _index, expert_detail_opinions in enumerate(expert_opinions_):
                        print(f"in DS_Evidence, 当前专家{_index} 给出各节点间关系共 {len(expert_detail_opinions)} 条")
                    print(f"in DS_Evidence 本图中总共 {len(Graph)} 条边")

    print(f"in DS_Evidence 专家意见已经收集完毕，开始融合")
    # 融合所有专家意见计算最终置信度
    result = calculate_edge_confidence(Graph, expert_opinions_, options_)

    return result, expert_opinions_



def D_S_Experts_human(
        Graph: list,
        options_: list,
        expert_opinions_: list):
    """
    使用D-S Experts对图进行修正
    expert_opinions_ = [
        {   # 专家1的判定结果
            "边名称": {
                "选项名称": 置信度,
                ...
            },
            ...
        },
        {   # 专家2的判定结果
            "边名称": {
                "选项名称": 置信度,
                ...
            },
            ...
        },
        ...
    ]
    :param expert_opinions_:
    :param nodes_name_lst_: 节点名称 eg: ['BirthAsphyxia', 'HypDistrib', 'HypoxiaInO2']
    :param Graph: 贝叶斯网络拓扑结构 eg:[('BirthAsphyxia', 'HypDistrib'),('BirthAsphyxia', 'HypoxiaInO2')]
    :param options_: 融合选项，例如['删除边', '保留边']
    :return: 各条已知边的置信度
    """
    print("in D_S_Experts_human, 调用人类专家知识进行构建")
    result = calculate_edge_confidence(Graph, expert_opinions_, options_)

    return result, expert_opinions_


def dsApproach(
        expertType,
        nodesNameLst,
        nodesInfoLst,
        Map2English,
        model,
        Debug=False,
        refresh=True,
        expertList=None,
        eps=0.4,
        aim=None,
):
    """
    根据专家类型（LLM或人类）对图模型中的边进行删除或保留操作。

    参数:
    - expertType: str, 专家类型，可选值为 'llm' 或 'human'。
    - nodesNameLst: list, 节点名称列表。
    - nodesInfoLst: list, 节点信息列表。
    - Map2English: dict, 将节点名称映射为英文的字典。
    - model: 图模型对象，包含图的边信息。
    - Debug: bool, 是否开启调试模式，默认为 False。
    - refresh: bool, 是否刷新专家列表，默认为 True。

    返回值:
    - model_new: 修改后的图模型对象，根据专家意见删除或保留了部分边。

    异常:
    - ValueError: 如果 expertType 不是 'llm' 或 'human'，则抛出异常。
    """
    options = ['删除边', '保留边']
    cache_detail_pth = os.path.join(pthcfg.cache_path, "DS_Experts_detail_cache.json")

    if expertType == 'llm':
        assert expertList is not None, "expertList must be not None when expertType is llm"
        assert aim is not None, "aim must be not None when expertType is llm"
        # 使用LLM专家进行D-S证据理论处理

        DS_result, expert_opinions_ = D_S_Experts(
            nodes_name_lst_=[Map2English[i] for i in nodesNameLst],
            nodes_info_list=nodesInfoLst,
            Graph=list(model.edges()),
            options_=options,
            debug=Debug,
            fresh=refresh,
            expert_list=expertList,
            aim=aim
        )
        model_new = model.copy()
        # 根据D-S结果修改图模型
        for edge in DS_result:
            if DS_result[edge]['保留边'] < eps:
                model_new.remove_edge(edge[0], edge[1])
            elif DS_result[edge]['保留边'] > eps > DS_result[edge]['删除边']:
                model_new.add_edge(edge[0], edge[1])

        return model_new, expert_opinions_, DS_result

    elif expertType == 'human':
        # 加载人类专家意见
        if aim == "Victim":
            opinions = json.load(open(os.path.join(pthcfg.assets_path, 'OpiniondataVicitim.json'), 'r'))
        else:
            opinions = json.load(open(os.path.join(pthcfg.assets_path, 'OpiniondataCrime.json'), 'r'))
        # 使用人类专家进行D-S证据理论处理
        DS_result, expert_opinions_ = D_S_Experts_human(
            Graph=list(model.edges()),
            options_=options,
            expert_opinions_=opinions
        )
        model_new = model.copy()
        # 根据D-S结果修改图模型
        for edge in DS_result:
            if DS_result[edge]['保留边'] < eps < DS_result[edge]['删除边']:
                model_new.remove_edge(edge[0], edge[1])
                print(f"in DS_Evidence, 边 {edge} 被删除")
            if DS_result[edge]['删除边'] < eps < DS_result[edge]['保留边']:
                model_new.add_edge(edge[0], edge[1])
                print(f"in DS_Evidence, 边 {edge} 被保留")
        return model_new, expert_opinions_, DS_result

    else:
        raise ValueError("expertType must be llm or human")


if __name__ == '__main__':
    # 示例输入
    edges = [('BirthAsphyxia', 'HypDistrib'),
             ('BirthAsphyxia', 'HypoxiaInO2'),
             ('HypDistrib', 'DuctFlow'),
             ('HypDistrib', 'LowerBodyO2'),
             ('HypDistrib', 'Disease'),
             ('HypDistrib', 'HypoxiaInO2'),
             ('HypoxiaInO2', 'RUQO2'),
             ('HypoxiaInO2', 'CardiacMixing'),
             ('HypoxiaInO2', 'LowerBodyO2'),
             ('HypoxiaInO2', 'Disease'),
             ('HypoxiaInO2', 'ChestXray'),
             ('HypoxiaInO2', 'LVHreport'),
             ('CO2', 'CO2Report'),
             ('CO2', 'ChestXray'),
             ('CO2', 'Grunting'),
             ('ChestXray', 'XrayReport'),
             ('ChestXray', 'LungParench'),
             ('ChestXray', 'Disease'),
             ('ChestXray', 'LungFlow'),
             ('ChestXray', 'Grunting'),
             ('ChestXray', 'LVHreport'),
             ('Grunting', 'GruntingReport'),
             ('Grunting', 'LungParench'),
             ('Grunting', 'Sick'),
             ('LVHreport', 'LVH'),
             ('LVHreport', 'Disease'),
             ('Disease', 'CardiacMixing'),
             ('Disease', 'DuctFlow'),
             ('Disease', 'LungFlow'),
             ('Disease', 'Age'),
             ('Disease', 'LungParench'),
             ('Disease', 'LVH'),
             ('Disease', 'Sick'),
             ('Age', 'Sick'),
             ('LungParench', 'LungFlow')]

    nodes_name_lst = ['BirthAsphyxia',
                      'Disease',
                      'Age',
                      'LVH',
                      'DuctFlow',
                      'CardiacMixing',
                      'LungParench',
                      'LungFlow',
                      'Sick',
                      'HypDistrib',
                      'HypoxiaInO2',
                      'CO2',
                      'ChestXray',
                      'Grunting',
                      'LVHreport',
                      'LowerBodyO2',
                      'RUQO2',
                      'CO2Report',
                      'XrayReport',
                      'GruntingReport']
    nodes_info_lst = ['表示婴儿出生时血液中氧气不足的情况。',
                      '表示婴儿高铁血红蛋白血症。',
                      '表示婴儿出现疾病时的年龄。',
                      '表示左心室肥厚。',
                      '表示动脉导管中的血流。',
                      '表示氧合血和脱氧血的混合。',
                      '表示肺部血管的状态。',
                      '表示肺部血流量低。',
                      '表示存在疾病。',
                      '表示身体各部位均匀分布的低氧区域。',
                      '表示吸氧时的低氧状态。',
                      '表示体内二氧化碳水平。',
                      '表示进行了胸部X光检查。',
                      '表示婴儿发出的哼声。',
                      '表示有关左心室肥厚的报告。',
                      '表示下身的氧含量。',
                      '表示右大腿前侧肌肉的氧含量。',
                      '表示报告血液中高水平二氧化碳的文件。',
                      '表示肺部过度充血的X光报告。',
                      '表示有关婴儿发出哼声的报告。']

    print(D_S_Experts(nodes_name_lst, nodes_info_lst, edges, ['删除边', '保留边'], 0, 5, 0))
