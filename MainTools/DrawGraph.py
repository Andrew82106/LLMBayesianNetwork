import matplotlib.pyplot as plt
import networkx as nx


def visualize_bayesian_network(bn_structure_, savepath, figsize=(12, 12)):
    """
    可视化贝叶斯网络
    :param bn_structure_: 贝叶斯网络结构，格式为列表的元组，例如 [('A', 'B'), ('B', 'C')]
    :param savepath: 图像保存路径
    :param figsize: 图像大小，默认为 (12, 12)
    """
    # 创建有向图对象
    model = nx.DiGraph(bn_structure_)

    # 计算节点和边的数量
    num_nodes = len(model.nodes)

    # 获取节点的层次信息
    def get_levels(G):
        levels_ = {node_: 0 for node_ in G.nodes}
        for node__ in G.nodes:
            if not list(G.predecessors(node__)):
                levels_[node__] = 0
            else:
                levels_[node__] = max([levels_[pred] for pred in G.predecessors(node__)]) + 1
        return levels_

    levels = get_levels(model)

    # 将层级信息设置为节点的属性
    for node, level in levels.items():
        model.nodes[node]['level'] = level

    # 使用 multipartite_layout 布局
    pos = nx.multipartite_layout(model, subset_key="level", align='horizontal')

    # 调整节点大小和颜色
    node_sizes = [300] * num_nodes  # 默认节点大小
    node_colors = ['skyblue'] * num_nodes  # 默认节点颜色

    # 调整字体大小
    font_size = max(8, 15 - int(num_nodes / 10))

    # 设置图像大小
    plt.figure(figsize=figsize)

    # 绘制图形
    nx.draw(model, pos, with_labels=True, node_size=node_sizes, node_color=node_colors, font_size=font_size,
            font_weight='bold')
    plt.title('Bayesian Network Structure')

    # 保存图像
    plt.savefig(savepath)
    plt.close()


def visualize_multiple_bayesian_networks(bn_structures, savepath, titles, figsize=(24, 24)):
    """
    可视化多个贝叶斯网络结构到2x2子图布局中

    参数:
    - bn_structures: list, 包含四个贝叶斯网络结构的列表，每个结构格式为元组列表
    - savepath: str, 图像保存路径
    - titles: list, 四个子图的标题列表
    - figsize: tuple, 整体图像尺寸，默认(24,24)
    """
    # 创建2x2子图画布
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs = axs.ravel()

    # 定义公共样式参数
    NODE_SIZE = 800
    NODE_COLOR = 'lightblue'
    FONT_WEIGHT = 'bold'

    for idx, (bn_structure, title) in enumerate(zip(bn_structures, titles)):
        model = nx.DiGraph(bn_structure)

        # 新增：计算并设置level属性（修复关键错误）
        levels = {node: 0 for node in model.nodes}
        for node in model.nodes:
            predecessors = list(model.predecessors(node))
            if predecessors:
                levels[node] = max(levels[p] for p in predecessors) + 1
        for node, level in levels.items():  # 设置level属性到节点
            model.nodes[node]['level'] = level  # 关键修复行

        pos = nx.multipartite_layout(model, subset_key="level", align='horizontal')
        num_nodes = len(model.nodes)
        font_size = max(8, 18 - int(num_nodes / 8))

        nx.draw(model, pos, ax=axs[idx], with_labels=True,
                node_size=NODE_SIZE, node_color=NODE_COLOR,
                font_size=font_size, font_weight=FONT_WEIGHT)

        axs[idx].set_title(title, fontsize=20)

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()


def visualize_multiple_bayesian_networks1(bn_structures, savepath, titles, figsize=(24, 94)):
    """
    可视化多个贝叶斯网络结构到2x10子图布局中

    参数:
    - bn_structures: list, 包含四个贝叶斯网络结构的列表，每个结构格式为元组列表
    - savepath: str, 图像保存路径
    - titles: list, 四个子图的标题列表
    - figsize: tuple, 整体图像尺寸，默认(24,24)
    """
    # 创建2x2子图画布
    fig, axs = plt.subplots(10, 2, figsize=figsize)
    axs = axs.ravel()

    # 定义公共样式参数
    NODE_SIZE = 800
    NODE_COLOR = 'lightblue'
    FONT_WEIGHT = 'bold'

    for idx, (bn_structure, title) in enumerate(zip(bn_structures, titles)):
        model = nx.DiGraph(bn_structure)

        # 新增：计算并设置level属性（修复关键错误）
        levels = {node: 0 for node in model.nodes}
        for node in model.nodes:
            predecessors = list(model.predecessors(node))
            if predecessors:
                levels[node] = max(levels[p] for p in predecessors) + 1
        for node, level in levels.items():  # 设置level属性到节点
            model.nodes[node]['level'] = level  # 关键修复行

        pos = nx.multipartite_layout(model, subset_key="level", align='horizontal')
        num_nodes = len(model.nodes)
        font_size = max(8, 18 - int(num_nodes / 8))

        nx.draw(model, pos, ax=axs[idx], with_labels=True,
                node_size=NODE_SIZE, node_color=NODE_COLOR,
                font_size=font_size, font_weight=FONT_WEIGHT)

        axs[idx].set_title(title, fontsize=20)

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # 示例网络结构
    bn_structure = [('BirthAsphyxia', 'LVH'), ('LVH', 'DuctFlow'), ('LVH', 'LungParench'), ('Disease', 'CardiacMixing'), ('CardiacMixing', 'LungParench'), ('CardiacMixing', 'LungFlow'), ('LungParench', 'HypDistrib'), ('LungFlow', 'HypDistrib'), ('LungFlow', 'CO2'), ('LungFlow', 'Grunting'), ('LungFlow', 'HypoxiaInO2'), ('HypDistrib', 'LowerBodyO2'), ('HypDistrib', 'RUQO2'), ('Sick', 'Age'), ('Sick', 'ChestXray'), ('Sick', 'LVHreport'), ('Sick', 'CO2Report'), ('Sick', 'XrayReport'), ('Sick', 'GruntingReport'), ('Sick', 'HypDistrib')]
    visualize_bayesian_network(bn_structure, 'test.png', figsize=(12, 12))
