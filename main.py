from datetime import datetime
import pickle
from MainTools.pthcfg import *

pthcfg = PathConfig()

from MainTools.utils import *
from MainTools.Expert import expertGenerator
from MainTools.K2_algorithm import k2Process
from MainTools.DrawGraph import visualize_bayesian_network
from MainTools.ISM_Approach import ismApproach
from MainTools.DS_Evidence import dsApproach


def main(
        REFRESH,
        DS_REFRESH,
        ISM_REFRESH,
        REFREFRESH_ISM_EXPERT,
        REFREFRESH_DS_EXPERT,
        ISM_EPS,
        DS_EPS,
        NUM_OF_EXPERT,
        LLM_TYPE,
        KNOWLEDGE_INSERT_METHOD,
        DEBUG
):
    """
    主函数，用于处理不同目标（Criminal, Victim）和专家类型（human, llm）的图模型生成与比较。

    该函数的主要流程如下：
    1. 遍历目标列表和专家类型列表，加载数据并生成ISM结果。
    2. 使用K2算法处理ISM结果，生成贝叶斯网络模型。
    3. 可视化并保存贝叶斯网络结构。
    4. 使用DS算法进一步处理模型，生成DS模型。
    5. 可视化并保存DS模型结构。
    6. 计算并输出不同目标下，llm与human生成的模型之间的F1分数。
    """
    if REFRESH:
        DS_REFRESH = 1
        ISM_REFRESH = 1
        REFREFRESH_ISM_EXPERT = 1
        REFREFRESH_DS_EXPERT = 1


    # 定义目标和专家类型列表
    aimList = ['Victim', "Criminal"]
    expertTypeList = ['llm', 'human']
    print("in main function: 初始化专家列表")

    if REFREFRESH_ISM_EXPERT or REFREFRESH_DS_EXPERT:
        print("in main function: 刷新专家列表")
        cleanExpertCache()
    flag = False
    while not flag:
        try:
            ismExpertList = expertGenerator(
                Debug=DEBUG,
                Mode=KNOWLEDGE_INSERT_METHOD,
                llmType=LLM_TYPE,
                numOfExpert=NUM_OF_EXPERT,
                refresh=REFREFRESH_ISM_EXPERT,
                stage='ism'
            )
            flag = True
        except Exception as e:
            print("in main function: 刷新专家列表失败，重试中")
            print(f"错误信息：{e}")
            continue

    flag = False
    while not flag:
        try:
            DSExpertList = expertGenerator(
                Debug=DEBUG,
                Mode=KNOWLEDGE_INSERT_METHOD,
                llmType=LLM_TYPE,
                numOfExpert=NUM_OF_EXPERT,
                refresh=REFREFRESH_DS_EXPERT,
                stage='ds'
            )
            flag = True
        except:
            print("in main function: 刷新专家列表失败，重试中")
            continue

    print("in main function: 专家列表初始化完毕")

    labData = {
        "parameter": {
            "DS_REFRESH": DS_REFRESH,
            "NUM_OF_EXPERT": NUM_OF_EXPERT,
            "LLM_TYPE": LLM_TYPE,
            "KNOWLEDGE_INSERT_METHOD": KNOWLEDGE_INSERT_METHOD,
            "DEBUG": DEBUG,
            "ISM_REFRESH": ISM_REFRESH,
            "ISM_EPS": ISM_EPS,
            "DS_EPS": DS_EPS
        },
        "result": {
            "ismResult": {},
            "dsResult": {}
        },
        "processData": {
            "ismDetailResult": {},
            'K2Model': {},
            "dsDetailResult": {}
        }
    }

    # 用于存储不同目标和专家类型生成的图模型
    graphs = {}

    # 遍历所有目标和专家类型组合
    for expertType in expertTypeList:
        for aim in aimList:
            # 加载数据，获取节点名称、节点信息、映射关系等
            nodesNameLst, nodesInfoLst, Map, Map2English = dataLoader(aim=aim)

            # 初始化专家列表
            ismExpertList = cleanExpertInstanceExtraKnowledge(ismExpertList)
            DSExpertList = cleanExpertInstanceExtraKnowledge(DSExpertList)
            print("in main function: 使用ISM方法生成ISM结果")
            # 使用ISM方法生成ISM结果
            ismResult, raw_connections = ismApproach(
                nodesNameLst=nodesNameLst,
                nodesInfoLst=nodesInfoLst,
                aim=aim,
                expertType=expertType,
                Debug=DEBUG,
                expert_list=ismExpertList,
                refresh=ISM_REFRESH,
                eps=ISM_EPS
            )
            # labData['result']['ismResult'][str(aim) + str(expertType)] = ismResult
            labData['result']['ismResult'] = ismResult
            # labData['processData']['ismDetailResult'][str(aim) + str(expertType)] = raw_connections
            labData['processData']['ismDetailResult'] = raw_connections

            print("ISM方法生成ISM结果完毕")
            # 使用K2算法处理ISM结果，生成贝叶斯网络模型
            if DEBUG:
                print("开始调用K2算法生成网络结构")
            model = k2Process(Map, Map2English, ismResult[str(aim) + str(expertType)], os.path.join(pthcfg.database_path, 'bayesian_victim_and_others_filled.csv'))
            visualize_bayesian_network(bn_structure_=list(model.edges()),
                                       savepath=os.path.join(pthcfg.log_pth, f"K2_{aim}:{expertType}.png"))
            labData['processData']['K2Model'][str(aim) + str(expertType)] = list(model.edges())

            # 使用DS算法进一步处理模型，生成DS模型
            print("开始调用DS算法生成网络结构")
            DS_model, ExpertOpinions, DS_result = dsApproach(
                expertType=expertType,
                nodesNameLst=nodesNameLst,
                nodesInfoLst=nodesInfoLst,
                Map2English=Map2English,
                model=model,
                Debug=DEBUG,
                refresh=DS_REFRESH,
                eps=DS_EPS,
                expertList=DSExpertList,
                aim=aim
            )
            labData['result']['dsResult'][str(aim) + str(expertType)] = DS_model
            labData['processData']['dsDetailResult'][str(aim) + str(expertType)] = {
                "ExpertOpinions": ExpertOpinions,
                "DS_funsion_result": DS_result
            }
            print("DS算法生成网络结构完毕")
            # 可视化并保存DS模型结构
            visualize_bayesian_network(bn_structure_=list(DS_model.edges()),
                                       savepath=os.path.join(pthcfg.log_pth, f"DS_{aim}:{expertType}.png"))

            # 将生成的模型存储在graphs字典中
            graphs[aim + expertType] = [model, DS_model]

    # 计算并输出Criminal目标下，llm与human生成的DS模型之间的F1分数
    CriminalF1 = calc_graph_f1(list(graphs['Criminalllm'][1].edges()), list(graphs['Criminalhuman'][1].edges()))
    print(f"Criminal final llm with human: F1 = {CriminalF1}")

    # 计算并输出Victim目标下，llm与human生成的DS模型之间的F1分数
    VictimF1 = calc_graph_f1(list(graphs['Victimllm'][1].edges()), list(graphs['Victimhuman'][1].edges()))
    print(f"Victim final llm with human: F1 = {VictimF1}")

    # 计算并输出Criminal目标下，llm与human生成的原始模型之间的F1分数
    CriminalRawF1 = calc_graph_f1(list(graphs['Criminalllm'][0].edges()), list(graphs['Criminalhuman'][0].edges()))
    print(f"Criminal raw llm with human: F1 = {CriminalRawF1}")

    # 计算并输出Victim目标下，llm与human生成的原始模型之间的F1分数
    VictimRawF1 = calc_graph_f1(list(graphs['Victimllm'][0].edges()), list(graphs['Victimhuman'][0].edges()))
    print(f"Victim raw llm with human: F1 = {VictimRawF1}")

    labData['result']['Graph'] = graphs
    # save with pickle
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(pthcfg.final_output_path, f'labData[time={date}].pkl'), 'wb') as f:
        pickle.dump(labData, f)



if __name__ == '__main__':
    ###############################################DASHBOARD#################################################
    DS_REFRESH_ = 0
    ISM_REFRESH_ = 0
    REFREFRESH_ISM_EXPERT_ = 0
    REFREFRESH_DS_EXPERT_ = 0
    ISM_EPS_ = 4
    DS_EPS_ = 0.15
    NUM_OF_EXPERT_ = 10
    # LLM_TYPE_ = "ChatGLM4Flash"
    # LLM_TYPE_ = "Qwen"
    # LLM_TYPE_ = "QWQ"
    LLM_TYPE_ = "QWQ_plus"
    KNOWLEDGE_INSERT_METHOD_ = "DEFAULT"
    # KNOWLEDGE_INSERT_METHOD_ = "ASSERT"
    # KNOWLEDGE_INSERT_METHOD_ = "DESCRIBE"
    # KNOWLEDGE_INSERT_METHOD_ = "RELATIONSHIP"
    DEBUG_ = True
    REFRESH_ = False
    ###############################################DASHBOARD#################################################

    main(
        REFRESH=REFRESH_,
        DS_REFRESH=DS_REFRESH_,
        ISM_REFRESH=ISM_REFRESH_,
        REFREFRESH_ISM_EXPERT=REFREFRESH_ISM_EXPERT_,
        REFREFRESH_DS_EXPERT=REFREFRESH_DS_EXPERT_,
        ISM_EPS=ISM_EPS_,
        DS_EPS=DS_EPS_,
        NUM_OF_EXPERT=NUM_OF_EXPERT_,
        LLM_TYPE=LLM_TYPE_,
        KNOWLEDGE_INSERT_METHOD=KNOWLEDGE_INSERT_METHOD_,
        DEBUG=DEBUG_
    )
