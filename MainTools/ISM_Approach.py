from pthcfg import *

pthcfg = PathConfig()

import time
import numpy as np
import networkx as nx
from Expert import *
from DrawGraph import visualize_bayesian_network


def has_cycle(relationship_matrix):
    """
    检查图中是否有环
    :param relationship_matrix:  二维的影响因素关系矩阵
    :return: bool，有无环
    """
    G = nx.DiGraph()

    # 构建图，假设relationship_matrix是影响矩阵
    size = relationship_matrix.shape[0]
    for i in range(size):
        for j in range(size):
            if relationship_matrix[i][j] == 1:
                G.add_edge(i, j)

    # 检测图中是否有环
    return nx.is_directed_acyclic_graph(G) is False


def transitive_closure(matrix):
    """
    传递闭包
    :param matrix: 二维的影响因素关系矩阵
    :return: 对Matrix的传递闭包结果
    """
    assert has_cycle(matrix) is False, "输入矩阵存在环，无法计算传递闭包"
    size = matrix.shape[0]
    result = np.copy(matrix)
    for k in range(size):
        for i in range(size):
            for j in range(size):
                result[i, j] = result[i, j] or (result[i, k] and result[k, j])
    return result


def find_cycles(relationship_matrix):
    """
    查找图中的所有环
    :param relationship_matrix:  二维的影响因素关系矩阵
    :return: list of lists，所有环的列表，每个环是一个节点的列表
    """
    G = nx.DiGraph()

    # 构建图，假设relationship_matrix是影响矩阵
    size = relationship_matrix.shape[0]
    for i in range(size):
        for j in range(size):
            if relationship_matrix[i][j] == 1:
                G.add_edge(i, j)

    # 查找图中的所有环
    cycles = list(nx.simple_cycles(G))
    return cycles


def destory_circiel(matrix):
    """
    破坏环
    :param matrix: 二维的影响因素关系矩阵
    :return: 破坏环的结果
    """
    # 将matrix中所有行和列相同的值都设为0
    for i in range(matrix.shape[0]):
        matrix[i][i] = 0
    cycles = find_cycles(matrix)
    while len(cycles):
        # print(f"图中存在环: {cycles}")
        node0 = cycles[0][0]
        node1 = cycles[0][0] if len(cycles[0]) == 1 else cycles[0][1]
        matrix[node0, node1] = 0
        # matrix[node1, node0] = 0
        cycles = find_cycles(matrix)
        # print(f'There are {len(cycles)} cycles in G now')
    print(f'in destory_circiel 去环完成')
    return matrix


def topological_sort(matrix) -> list:
    """
    拓扑排序
    :param matrix: 二维的影响因素关系矩阵
    :return: 排序结果
    """
    # 将矩阵转换为有向图
    assert has_cycle(matrix) is False, "输入矩阵存在环，无法拓扑排序"
    size = matrix.shape[0]
    edges = []
    for i in range(size):
        for j in range(size):
            if matrix[i][j] == 1:
                edges.append((i, j))
    # 拓扑排序
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # 计算拓扑排序
    topological_sort_res = list(nx.topological_sort(G))
    return topological_sort_res


def convert_matrix_to_bn(matrix, nodes_name_lst: list):
    """
    将矩阵转换为贝叶斯网络结构
    :param matrix: 二维的影响因素关系矩阵
    :param nodes_name_lst: 节点名称列表
    :return: 贝叶斯网络结构
    """
    bn_structures = []
    size = matrix.shape[0]
    for i in range(size):
        for j in range(size):
            if matrix[i][j] == 1:
                bn_structures.append((nodes_name_lst[i], nodes_name_lst[j]))
    return bn_structures


def ISM_Experts(
        nodes_name_lst: list,
        nodes_info_list: list,
        debug=False,
        expert_list=None
):
    """
    ISM算法全流程
    :param expert_list: 专家列表，若为None则使用默认的专家
    :param nodes_name_lst: 节点名称列表
    :param nodes_info_list: 节点信息列表
    :param debug: 是否开启调试模式
    :return: 拓扑排序结果
    """

    # 先做一个map
    name_to_index = {nodes_name_lst[i]: i for i in range(len(nodes_name_lst))}
    if expert_list is None:
        raise Exception("expert_list is None")
    m = np.zeros((len(nodes_name_lst), len(nodes_name_lst)))
    raw_connections = []
    for index, expert in enumerate(expert_list):
        # if debug:
        #     print(f"waiting for LLM response of generating conections (number {index}/{numofExpert}, LLM_name: {expert.llm_name})")
        connections = generate_conections(
            nodes_name_lst=nodes_name_lst,
            nodes_info_list=nodes_info_list,
            llm_=expert
        )
        raw_connections.append(connections)
        if debug:
            print(f"in ISM_Experts, LLM_name: {expert.llm_name} 生成的连接为:{connections}")
        for connection in connections:
            try:
                m[name_to_index[connection[0]], name_to_index[connection[1]]] = 1
            except:
                print(f"connection {connection} is illegal")
                print(f'name to index: {name_to_index}')
                print(f'connections: {connections}')
                print(f"nodes_info_list: {nodes_info_list}")
                print(f"nodes_name_lst: {nodes_name_lst}")
                print("ERROR：！！！connection is illegal")
                # raise Exception("connection is illegal")
                continue

        time.sleep(0.5)

    if debug:
        if has_cycle(m):
            print(f"存在环:{find_cycles(m)}，无法进行拓扑排序, 开始尝试破坏环")

    m = destory_circiel(m)

    assert has_cycle(m) is False, "存在环，无法进行拓扑排序"

    m1 = transitive_closure(m)

    topo_result = topological_sort(m1)

    topo_result_with_name = [nodes_name_lst[i] for i in topo_result]

    for i in range(len(nodes_name_lst)):
        if i not in topo_result:
            topo_result_with_name.append(nodes_name_lst[i])

    return topo_result_with_name, raw_connections


def checkISMData(connection_opinion, nodes_name_lst, debug=False):
    legalConnection = [[] for _ in range(len(connection_opinion))]
    for index, expertOpinion in enumerate(connection_opinion):
        for connection in expertOpinion:
            if connection[0] not in nodes_name_lst or connection[1] not in nodes_name_lst:
                continue
            elif connection[0] == connection[1]:
                continue
                # if debug:
                # print(f"in ISM_Approach checkISMData, connection_opinion中的节点名称:{connection[0], connection[1]}不在nodes_name_lst中，进行丢弃")
            else:
                legalConnection[index].append(connection)
    return legalConnection


def ISM_Experts_human(
        connection_opinion_raw: list,
        nodes_name_lst: list,
        debug=False,
        eps=5,
        aim=None
):
    """
    ISM算法全流程-人类专家
    :param eps:
    :param connection_opinion_raw: 专家给出的意见: [[(node1, node2, 1/0), (node3, node4, 1/0), ...], [], ...]
    :param nodes_name_lst: 节点名称列表
    :param debug: 是否开启调试模式
    :return: 拓扑排序结果
    """
    connection_opinion = checkISMData(connection_opinion_raw, nodes_name_lst, debug)
    ism_cache = []
    for connection_op in connection_opinion:
        ism_cache.append([])
        for connection_i in connection_op:
            ism_cache[-1].append([connection_i[0], connection_i[1]])

    # 先做一个map
    name_to_index = {nodes_name_lst[i]: i for i in range(len(nodes_name_lst))}

    m = np.zeros((len(nodes_name_lst), len(nodes_name_lst)))

    for connection_op in connection_opinion:
        for connection in connection_op:
            if connection[2] == 1:
                m[name_to_index[connection[0]], name_to_index[connection[1]]] += 1
    # 在eps以下的m中的值设为0，其他的设为1
    for i in range(len(m)):
        for j in range(len(m)):
            if m[i][j] < eps:
                m[i][j] = 0
            else:
                m[i][j] = 1

    if debug:
        if has_cycle(m):
            print(f"in ISM_Experts_human,存在环，无法进行拓扑排序，开始去环")

    m = destory_circiel(m)

    assert has_cycle(m) is False, "存在环，无法进行拓扑排序"

    m1 = transitive_closure(m)

    topo_result = topological_sort(m1)

    topo_result_with_name = [nodes_name_lst[i] for i in topo_result]

    # 把漏掉的节点也加入到结果中
    for i in range(len(nodes_name_lst)):
        if i not in topo_result:
            topo_result_with_name.append(nodes_name_lst[i])
    # topo_result_with_name.append("嫌疑人性别")
    if aim == 'Victim':
        topo_result_with_name = [
            "受害者收入水平",
            "受害者性别",
            "受害者年龄",
            "受害者受教育程度",
            "受害者职业",
            "发案日期特殊性",
            "嫌疑人和受害者联系方式",
            "发案城市经济水平",
            "损失金额",
            "是否主动报警",
            "恢复时间"
        ]
    elif aim == 'Criminal':
        topo_result_with_name = [
            "嫌疑人性别",
            "嫌疑人学历",
            "嫌疑人和受害者联系方式",
            "发案日期特殊性",
            "是否主动报警",
            "发案城市经济水平",
            "嫌疑人年龄",
            "嫌疑人职业",
            "嫌疑人诈骗能力",
            "损失金额",
            "恢复时间"
        ]
    return topo_result_with_name, ism_cache


def ismApproach(
        nodesNameLst,
        nodesInfoLst,
        aim='human',
        expertType="llm",
        Debug=False,
        expert_list=None,
        refresh=False,
        eps=5
):
    funsionType = str(aim) + str(expertType)

    # 初始化缓存文件路径
    cache_pth = os.path.join(pthcfg.cache_path, "ism_cache.json")
    cache_detail_pth = os.path.join(pthcfg.cache_path, "ism_detail_cache.json")
    ism_opinions_ = {}
    ism_detail_ = {}
    # 处理缓存文件：若需刷新则清空，否则加载已有数据
    if refresh:
        # 删除文件
        os.remove(cache_pth)
        # 将初始数据写入缓存文件
        with open(cache_pth, 'w') as f:
            json.dump(ism_opinions_, f)
        with open(cache_detail_pth, 'w') as f:
            json.dump(ism_detail_, f)

    if os.path.exists(cache_pth):
        # 加载已有缓存数据
        with open(cache_pth, 'r') as f:
            ism_opinions_ = json.load(f)
        with open(cache_detail_pth, 'r') as f:
            ism_detail_ = json.load(f)
        if funsionType in ism_opinions_:
            print(f"in ismApproach method: 目标{str(aim)}的{str(expertType)}意见已有本地缓存，因此直接读取，不经过ism算法")
            return ism_opinions_, ism_detail_
    elif not os.path.exists(cache_pth):
        # 创建初始缓存文件
        with open(cache_pth, 'w') as f:
            json.dump(ism_opinions_, f)

    if expertType == "llm":
        ism_opinions_[funsionType], ism_detail_[funsionType] = ISM_Experts(
            nodes_name_lst=nodesNameLst,
            nodes_info_list=nodesInfoLst,
            debug=Debug,
            expert_list=expert_list
        )
    elif expertType == 'human':
        expert_opinion_ = json.load(open(os.path.join(pthcfg.assets_path, 'expert_opinion.json'), 'r'))
        ism_opinions_[funsionType], ism_detail_[funsionType] = ISM_Experts_human(
            connection_opinion_raw=expert_opinion_,
            nodes_name_lst=nodesNameLst,
            debug=Debug,
            eps=eps,
            aim=aim
        )
    else:
        raise ValueError("expertType must be llm or human")
    # save cache
    with open(cache_pth, 'w') as f:
        json.dump(ism_opinions_, f)
        print(f"in ismApproach method: 目标{str(aim)}的{str(expertType)} ism意见已保存到本地缓存")
    with open(cache_detail_pth, 'w') as f:
        json.dump(ism_detail_, f)
        print(f"in ismApproach method: 目标{str(aim)}的{str(expertType)} ism细节已保存到本地缓存")
    return ism_opinions_, ism_detail_


if __name__ == '__main__':
    nodes_name_lsts = ['受害者住址',
                       '受害者职业',
                       '受害者性别',
                       '受害者年龄',
                       '受害者收入水平',
                       '受害者受教育程度',
                       '嫌疑人年龄',
                       '嫌疑人活动位置',
                       '嫌疑人学历',
                       '嫌疑人诈骗动机',
                       '嫌疑人诈骗能力',
                       '嫌疑人和受害者联系方式',
                       '发案时间段',
                       '发案日期特殊性',
                       '发案城市经济水平',
                       '损失金额',
                       '是否主动报警',
                       '恢复时间']

    nodes_info_lsts = ['表示受害者的居住地址。',
                       '表示受害者的就业情况或职业。',
                       '表示受害者的性别。',
                       '表示受害者的年龄。',
                       '表示受害者的收入水平，包括低、中、高。',
                       '表示受害者的学历，包括小学、初中、高中、大学、硕士研究生、博士生。',
                       '表示嫌疑人的年龄。',
                       '表示嫌疑人的活动地点。',
                       '表示嫌疑人的学历。',
                       '表示嫌疑人实施诈骗的动机。',
                       '衡量嫌疑人的诈骗能力，包括低、中、高三个等级。',
                       '表示嫌疑人和受害者之间的联系方式。',
                       '表示案件发生的时段，包括早晨、中午、下午、晚上、半夜、凌晨。',
                       '表示发案日期所接近的最近节假日的天数。',
                       '表示案件发生的城市的经济水平，包括低、中、高。',
                       '表示受害者被诈骗的金额，包括100以下，100到1000,1000到5000,5000到10000,10000到10 0000,10 0000 到100 0000,100 0000 以上。',
                       '表示受害者在案件中是否有主动报警。',
                       '表示受害者收到诈骗后生活恢复原水平的时间。']
    r = "/Users/andrewlee/Desktop/Projects/实验室/114项目/bayesian_data/expert_opinion.json"
    import json

    expert_opinion = json.load(open(r, 'r'))
    res = ISM_Experts_human(expert_opinion, nodes_name_lsts, debug=True)
    print(res)
