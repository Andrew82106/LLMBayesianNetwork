import random
import networkx as nx
import pandas as pd
from pgmpy.estimators import K2Score
from pgmpy.models import BayesianNetwork
import numpy as np


class K2:
    def __init__(self, data_, max_parents=None):
        """
        初始化K2算法。

        参数:
        data_ (pd.DataFrame): 数据集。
        max_parents (int, optional): 每个节点最大父节点数。默认为None，表示不限制。
        """
        self.data = data_
        self.node_order = None
        self.all_node_list = data_.columns.tolist()
        self.score_func = K2Score(data_)
        self.model = BayesianNetwork()
        self.max_parents = max_parents if max_parents is not None else float('inf')

    def get_order_from_raw_order(self, raw_order: list):
        """
        从原始的节点顺序列表中获取正确的节点顺序。
        输入Be like：[['Sick', 'Disease'], ['LungFlow', 'HypoxiaInO2'], ['CardiacMixing', 'HypDistrib'], ['CO2', 'CO2Report'], ['Grunting', 'GruntingReport'], ['ChestXray', 'XrayReport']]
        输出Be like：['Sick', 'Disease', 'LungFlow', 'HypoxiaInO2', 'CardiacMixing', 'CO2', 'CO2Report', 'Grunting', 'GruntingReport', 'ChestXray', 'XrayReport']
        :param raw_order: 原始的节点顺序列表
        :return: 整理后的节点顺序列表
        """
        # 输入中，['A', 'B']表示A到B有一条有向边
        # 基于输入数据建立有向图
        graph = nx.DiGraph()
        for edge in raw_order:
            graph.add_edge(edge[0], edge[1])

        # 检测图中是否存在环
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("输入的图中存在环，无法进行拓扑排序")

        nodes_layers = []
        while graph.nodes:
            # 遍历图中所有节点，获得图中出度为0的节点
            layer_cur = [node for node in graph.nodes if graph.out_degree(node) == 0]
            if not layer_cur:
                raise ValueError("输入的图中存在环，无法进行拓扑排序")

            # 将layer_cur中的节点从图中删除
            graph.remove_nodes_from(layer_cur)
            nodes_layers.append(layer_cur)

        self.node_order = []
        for layer in nodes_layers:
            # 使用random.sample创建一个新的随机排列
            self.node_order.extend(random.sample(layer, len(layer)))

        for nodes in self.all_node_list:
            if nodes not in self.node_order:
                self.node_order.append(nodes)

    def set_node_order(self, order: list):
        """
        设置节点顺序列表。
        """
        self.node_order = order

    def get_standard_node_oder(self, order: list):
        """
        将原始的节点顺序列表转换为标准节点顺序列表。
        """
        self.node_order = order

    def learn_structure(self) -> BayesianNetwork:
        """
        使用K2算法学习贝叶斯网络的结构。
        标准K2算法是一种贪心搜索方法，按照预定义的节点顺序，
        为每个节点选择最优的父节点集合。

        返回:
        BayesianNetwork: 学习到的贝叶斯网络结构。
        """
        assert self.node_order is not None, "请先设置节点顺序"
        
        # 初始化一个空的贝叶斯网络
        self.model = BayesianNetwork()
        for node in self.all_node_list:
            self.model.add_node(node)
            
        # 按照节点顺序处理每个节点
        for i, node in enumerate(self.node_order):
            # 当前节点的潜在父节点集合（排序中位于该节点之前的所有节点）
            potential_parents = self.node_order[:i]
            
            # 当前节点的最优父节点集合
            parents = []
            
            # 记录当前最高分数
            best_score = self.score_func.local_score(node, parents)
            
            # 标记是否继续添加父节点
            continue_adding = True
            
            # 当可以继续添加父节点且有潜在父节点可选时
            while continue_adding and len(parents) < self.max_parents and potential_parents:
                # 初始化最优父节点和最高分数增益
                best_parent = None
                best_new_score = best_score
                
                # 尝试每个潜在父节点
                for parent in potential_parents:
                    if parent not in parents:
                        # 计算添加该父节点后的分数
                        new_parents = parents + [parent]
                        new_score = self.score_func.local_score(node, new_parents)
                        
                        # 如果分数更高，更新最优父节点
                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_parent = parent
                
                # 如果找到了能提高分数的父节点，添加该父节点
                if best_parent is not None and best_new_score > best_score:
                    parents.append(best_parent)
                    best_score = best_new_score
                    # 从potential_parents中移除已添加的父节点
                    potential_parents.remove(best_parent)
                else:
                    # 没有找到能提高分数的父节点，停止添加
                    continue_adding = False
            
            # 为当前节点添加所有最优父节点
            for parent in parents:
                self.model.add_edge(parent, node)
        
        return self.model

    def score(self, node, parents):
        """
        计算节点的K2评分函数。

        参数:
        node (str): 当前节点。
        parents (list): 候选父节点列表。

        返回:
        float: 评分函数值。
        """
        return self.score_func.local_score(node, parents)


def k2Process(Map_, Map_2_English_, ismResult, csv_path):

    data_ = pd.read_csv(csv_path)
    # 将data中在Map的键值中的节点列抽取出来，其他的不要
    data_ = data_[[i for i in data_.columns if i in Map_]]
    # 然后将Map中的键值替换为Map中的值
    data_.rename(columns=Map_, inplace=True)
    # 然后将Map中的键值替换为Map_2_English中的值
    data_.rename(columns=Map_2_English_, inplace=True)
    # 然后将result中存在于Map中的值替换为Map_2_English中的值
    result0 = ismResult.copy()
    result = []
    for i in result0:
        if i in Map_2_English_:
            result.append(Map_2_English_[i])

    k2 = K2(data_)
    k2.get_standard_node_oder(result)
    model = k2.learn_structure()
    return model


if __name__ == '__main__':
    # 示例数据
    data = pd.DataFrame(
        np.random.randint(0, 2, size=(1000, 5)),  # 使用离散数据，K2更适合离散变量
        columns=list('ABCDE')
    )

    # 创建K2对象
    k2 = K2(data, max_parents=2)  # 限制每个节点最多有2个父节点
    
    # 设置节点顺序
    k2.set_node_order(['A', 'B', 'C', 'D', 'E'])
    
    # 学习贝叶斯网络结构
    model = k2.learn_structure()

    print("学习到的贝叶斯网络结构：")
    print(model.edges())