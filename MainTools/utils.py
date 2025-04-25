import json
from pgmpy.readwrite import BIFReader
from pgmpy.models.BayesianNetwork import BayesianNetwork
from pthcfg import *
import pickle as pkl


def readOutputPkl(filename):
    with open(filename, 'rb') as f:
        content = pkl.load(f)
    return content


def read_json_file(file_path):
    """
    读取JSON文件。
    参数:
    file_path (str): JSON文件路径。
    返回:
    dict: JSON文件中的数据。
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def write_json_file(file_path, data):
    """
    写入JSON文件。
    参数:
    file_path (str): JSON文件路径。
    data_ (dict): 要写入的数据。
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def define_bayesian_network_from_bif(file_path) -> BayesianNetwork:
    """
    从BIF文件中读取贝叶斯网络的结构。

    参数:
    file_path (str): BIF文件路径。

    返回:
    BayesianNetwork: 定义好的贝叶斯网络对象。
    """
    reader = BIFReader(file_path)
    model = reader.get_model()
    return model


def calc_graph_f1(graph_pred_, graph_true_):
    """
    计算图的F1分数。

    参数:
    graph_pred_ (list): 预测的图, example: [('BirthAsphyxia', 'HypDistrib'), ('BirthAsphyxia', 'HypoxiaInO2'), ('HypDistrib', 'DuctFlow')]
    graph_true_ (list): 真实的图, example: [('BirthAsphyxia', 'HypDistrib'), ('BirthAsphyxia', 'HypoxiaInO2'), ('HypDistrib', 'DuctFlow')]

    返回:
    float: 图的F1分数。
    """

    # 转换为集合，方便进行集合运算
    set_pred = set(graph_pred_)
    set_true = set(graph_true_)

    # 计算 TP, FP, FN
    tp = len(set_pred & set_true)  # 预测的边在真实图中存在
    fp = len(set_pred - set_true)  # 预测的边在真实图中不存在
    fn = len(set_true - set_pred)  # 真实图中的边没有被预测到

    # 计算精确度和召回率
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    # 计算F1分数
    if precision + recall == 0:
        return 0.0
    f1_score_ = 2 * (precision * recall) / (precision + recall)

    return f1_score_, precision, recall


def cleanExpertCache():
    cachePath = PathConfig().log_pth
    for file in os.listdir(cachePath):
        if "txt" in file:
            os.remove(os.path.join(cachePath, file))


def cleanExpertInstanceExtraKnowledge(expert_list):
    for expert in expert_list:
        expert.refresh_2_baseKnowledge()
    return expert_list


def dataLoader(aim="Criminal"):
    """
    返回节点信息列表
    :param aim:
    :return:
    """
    assert aim in ["Criminal", "Victim"], "aim must be 'Criminal' or 'Victim'"

    Map = {
        "Criminal": {
            "案发城市经济水平_code": "发案城市经济水平",
            '是否报警_code': "是否主动报警",
            "相关人恢复时间_code": "恢复时间",
            "金额损失_code": "损失金额",
            "相关人通信方式_code": "嫌疑人和受害者联系方式",
            "相关人学历_code": "嫌疑人学历",
            "相关人收入_code": "嫌疑人诈骗能力",
            "相关人年龄_code": "嫌疑人年龄",
            "相关人性别_code": "嫌疑人性别",
            "工作大类_code": "嫌疑人职业",
            "案发日期敏感度_code": "发案日期特殊性"
        },
        "Victim": {
            "案发城市经济水平_code": "发案城市经济水平",
            '是否报警_code': "是否主动报警",
            "相关人恢复时间_code": "恢复时间",
            "金额损失_code": "损失金额",
            "相关人通信方式_code": "嫌疑人和受害者联系方式",
            "相关人学历_code": "受害者受教育程度",
            "相关人收入_code": "受害者收入水平",  #
            "相关人年龄_code": "受害者年龄",
            "相关人性别_code": "受害者性别",  #
            "案发日期敏感度_code": "发案日期特殊性",
            "工作大类_code": "受害者职业"
        }
    }

    Map2English = {
        "Criminal": {
            "发案城市经济水平": "CityEconomy",
            '是否主动报警': "CallPolice",
            "恢复时间": "RecoverTime",
            "损失金额": "Loss",
            "嫌疑人和受害者联系方式": "CommunicationMethod",
            "嫌疑人学历": "SuspectEducation",
            "嫌疑人诈骗能力": "SuspectAbility",
            "嫌疑人年龄": "SuspectAge",
            "嫌疑人性别": "SuspectGender",
            "嫌疑人职业": "SuspectOccupation",
            "发案日期特殊性": "SpecialDate"
        },
        "Victim": {
            "发案城市经济水平": "CityEconomy",
            '是否主动报警': "CallPolice",
            "恢复时间": "RecoverTime",
            "损失金额": "Loss",
            "嫌疑人和受害者联系方式": "CommunicationMethod",
            "受害者受教育程度": "VictimEducation",
            "受害者收入水平": "VictimIncome",
            "受害者年龄": "VictimAge",
            "受害者性别": "VictimSex",
            "发案日期特殊性": "SpecialDate",
            "受害者职业": "VictimOccupation",
        }
    }

    nodeJson = read_json_file(PathConfig().nodes_info_json)
    allNode = [i['name'] for i in nodeJson]
    nodesNameLst = []
    nodesInfoLst = []

    for name in Map2English[aim]:
        if name not in allNode:
            raise ValueError("nodeJson中不存在%s" % name)

    for instance in nodeJson:
        if instance['name'] not in Map2English[aim]:
            continue
        nodesNameLst.append(instance['name'])
        nodesInfoLst.append(instance['description'])

    return nodesNameLst, nodesInfoLst, Map[aim], Map2English[aim]


if __name__ == '__main__':
    graph_pred = [('BirthAsphyxia', 'HypDistrib'), ('BirthAsphyxia', 'HypoxiaInO2'), ('HypDistrib', 'DuctFlow')]
    graph_true = [('BirthAsphyxia', 'HypDistrib'), ('HypDistrib', 'DuctFlow')]

    f1_score = calc_graph_f1(graph_pred, graph_true)
    print(f"F1 Score: {f1_score}")