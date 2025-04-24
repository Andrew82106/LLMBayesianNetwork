import json
import random
from LLM import Qwen, GLM, Deepseek
import time
from LLM.Meta_LLM import LargeLanguageModel
from pthcfg import PathConfig
import os
import pickle
import pandas as pd
pthcfg = PathConfig()


def init_raw_experts(llmName=None, debug=False, llmType='ChatGLM4Flash') -> LargeLanguageModel:
    if llmType == 'ChatGLM4Flash':
        llm_ = GLM.ChatGLM_Origin_Zhipu(llmName)
        if llmName is not None:
            llm_.llm_name = llmName
    elif llmType == 'Qwen':
        llm_ = Qwen.Qwen(llmName)
        if llmName is not None:
            llm_.llm_name = llmName
    elif llmType == 'QWQ':
        llm_ = Qwen.QWQ(llmName)
        if llmName is not None:
            llm_.llm_name = llmName
    elif llmType == 'QWQ_plus':
        llm_ = Qwen.QWQ_plus(llmName)
        if llmName is not None:
            llm_.llm_name = llmName
    elif llmType == 'Deepseek_R1':
        llm_ = Deepseek.Deepseek_R1(llmName)
        if llmName is not None:
            llm_.llm_name = llmName
    else:
        raise ValueError("llmType参数错误, llmType应该包含在如下列表中：[ChatGLM4Flash, Qwen, QWQ, QWQ_plus, Deepseek-R1]")
    llm_.init_log_pth(os.path.join(pthcfg.log_pth, f"{llmType}-{llm_.llm_name}.txt"))
    llm_.open_history_log()
    if debug:
        llm_.open_debug_mode()

    return llm_


def init_experts(debug=False, llmType='ChatGLM4Flash') -> LargeLanguageModel:
    experts_prompt = "背景：你是一个电信诈骗领域的专家，我们现在需要构建电信诈骗的贝叶斯网络，你接下来的回答要基于你电信诈骗的知识进行专业的回答。"
    # experts_prompt = "背景：你是一个医学领域的专家，你接下来的回答要基于你医学领域的知识进行专业的回答。"
    response_prompt = "如果你能理解，回复收到即可，不要回复多余的字符。"
    prompt_ = experts_prompt + response_prompt

    # time_now_ms = datetime.datetime.now().timestamp()
    # llm_ = Qwen.Qwen()
    # llm_.init_log_pth(os.path.join(pthcfg.log_pth, f"QwenLog{llm_.llm_name}.txt"))
    llm_ = init_raw_experts(debug=debug, llmType=llmType)

    response = llm_.response_only_text(llm_.generate_msg(prompt_))
    if llm_.debug_mode:
        print("in init_experts" + response)
    else:
        print(f"in init_experts, ISM_Approach: LLM_name: {llm_.llm_name} 已响应，响应长度为:{len(response)}")

    return llm_


def insert_knowledge(llm_: LargeLanguageModel, Mode="DEFAULT") -> LargeLanguageModel:
    """
    基于KB为大模型注入专家知识
    :param Mode:
    :param llm_:
    :return: 注入专家知识的大模型
    """
    # KB = pd.read_csv(os.path.join(pthcfg.kb_path, "ChildKB.csv"))
    # KB = pd.read_csv(os.path.join(pthcfg.kb_path, "child.csv"))
    # relations = []
    # for index, row in KB.iterrows():
    #     relation = f" {row['Subject']} {row['RelationShip']} {row['Object']}。"
    #     relations.append(relation)
    #
    # # 将所有关系合并为一段描述
    # Knowledge = "以下是从知识库中提取的相关知识描述，这些信息有助于你构建贝叶斯网络：\n" + "\n".join(relations)
    # response = llm_.response_only_text(llm_.generate_msg(Knowledge))
    # if llm_.debug_mode:
    #     print(response)
    # return llm_
    rKB = pthcfg.knowledge_base_json
    KB = json.load(open(rKB, 'r', encoding='utf-8'))
    assert Mode in ['DEFAULT', 'ASSERT', 'DESCRIBE', 'RELATIONSHIP'], "Mode参数错误"
    knowledge = []
    if Mode == 'DEFAULT':
        for key in KB:
            lenKnowledge = len(KB[key])
            chooseNum = random.randint(1, lenKnowledge)
            originalList = KB[key]
            random_seeds = time.time()
            random.seed(random_seeds)
            random.shuffle(originalList)
            for i in range(chooseNum):
                knowledge.append(originalList[i])
    else:
        lenKnowledge = len(KB[Mode])
        chooseNum = random.randint(1, lenKnowledge)
        originalList = KB[Mode]
        random_seeds = time.time()
        random.seed(random_seeds)
        random.shuffle(originalList)
        for i in range(chooseNum):
            knowledge.append(originalList[i])

    Knowledge = "以下是从知识库中提取的相关知识描述，这些信息有助于你构建贝叶斯网络：\n" + "\n".join(knowledge)
    response = llm_.response_only_text(llm_.generate_msg(Knowledge))
    llm_.expertKnowledge = llm_.chat_history
    if llm_.debug_mode:
        print(response)
    else:
        print("ISM_Experts: LLM_name: ", llm_.llm_name, "理解问题并回答，回答长度为：", len(response), "字")
    return llm_



def generate_conections(nodes_name_lst: list, nodes_info_list: list, llm_: LargeLanguageModel, random_seeds=0) -> list:
    """
    输入网络节点信息和解释，输出网络结构
    :param nodes_name_lst: 网络节点名称
    :param nodes_info_list: 网络节点信息
    :param llm_: LLM
    :param random_seeds: 随机种子，用于生成具有不同背景的专家   TODO：这一段还没写
    :return: 网络结构
    """
    merged_info = [nodes_name_lst[i] + ":" + nodes_info_list[i] for i in range(len(nodes_info_list))]
    task_prompt = "任务描述：现在我这里有很多和电信诈骗相关的概念，把你认为最相关的概念整理并且以固定的格式输出，不需要关注那些只存在间接影响的节点。"
    list_prompt = f"相关的概念和解释如下：{merged_info}。"
    structure_prompt = "你的输出应该按照固定的格式。比如，如果你认为A会直接影响B，C会直接影响D，则最后输出一个嵌套列表：[['A', 'B'] ,['C', 'D']]。特别注意，你本次的输出应该只有上述的这个列表，不要输出其他任何多余的字符。同时特别注意，你输出的节点应该一字不差的严格包含在我告知你的节点列表中。"

    prompt_ = task_prompt + list_prompt + structure_prompt
    if llm_.debug_mode:
        print('waiting for LLM response of generating conections')

    response1 = llm_.response_only_text(llm_.generate_msg(prompt_))

    if llm_.debug_mode:
        print(response1)


    return eval(response1)


def generate_relief(nodes_pair: list, nodes_info_dict: dict, llm_: LargeLanguageModel) -> float:
    """
    输入网络节点和信息解释，输出删除和保留每条边的置信度，用于D-S
    :param nodes_pair: 网络节点对 eg:['node1', 'node2']
    :param nodes_info_dict: 网络节点信息 eg:{'node1': '含义XXX', 'node2': '含义XXXX'}
    :param llm_: LLM
    :return:
    """
    task_prompt = f"任务描述：现在我这里有两个贝叶斯网络中的节点{nodes_pair[0]}和{nodes_pair[1]}，你需要基于我给的该节点的含义信息，给出你认为{nodes_pair[0]}到{nodes_pair[1]}这条单向边存在的置信度。"
    list_prompt = f"相关的概念和解释如下：{nodes_info_dict}。"
    structure_prompt = "特别注意！你的输出应该只返回一行一个0到1之间的小数作为你的评分。这个数字越接近1代表着你认为这条边越可能存在。除了这个小数以外不要生成任何其他的东西。"

    prompt_ = task_prompt + list_prompt + structure_prompt

    if llm_.debug_mode:
        print('waiting for LLM response of generating conections')

    response1 = llm_.response_only_text(llm_.generate_msg(prompt_))

    if llm_.debug_mode:
        print(response1)

    # assert isinstance(eval(response1), float), "LLM返回值不是float"

    # return eval(response1)

    cnt = 0

    result = None

    while cnt < 10:
        cnt += 1
        try:
            result = eval(response1)
            if isinstance(result, float):
                break
        except Exception as e:
            if llm_.debug_mode:
                print("generation structure failed, retrying... debug info:" + str(e))
            llm_.step_back()
            llm_.step_back()
            response1 = llm_.response_only_text(llm_.generate_msg(prompt_))

    if result is None:
        raise Exception("LLM generation failed")

    return result


def generateNewExpert(
        debug=False,
        Mode="DEFAULT",
        llmType="ChatGLM4Flash",
        numOfExpert=5,
        stage="ism",
        llmPath=None
):
    initExpertList = [init_experts(debug=debug, llmType=llmType) for _ in range(numOfExpert)]
    print(f"in generateNewExpert, finish initing expert list(len={len(initExpertList)})")
    expert_list = [insert_knowledge(initExpertList[i], Mode=Mode) for i in range(numOfExpert)]
    print(f"in generateNewExpert, finish inserting knowledge to expert list(len={len(expert_list)})")
    expert_list_dump = [llm.dump_data() for llm in expert_list]
    with open(llmPath, 'wb') as f:
        pickle.dump(expert_list_dump, f)
    print(f"in expertGenerator method: 生成本地{stage}专家缓存文件，共{len(expert_list)}个专家")
    return expert_list


def expertGenerator(
        Debug=False,
        Mode="DEFAULT",
        llmType="ChatGLM4Flash",
        numOfExpert=5,
        refresh=False,
        stage="ism"
):
    assert stage in ['ism', 'ds'], "stage must be 'ism' or 'ds'"
    llmPath = os.path.join(pthcfg.cache_path, str(stage) + '.pkl')
    if os.path.exists(llmPath) and not refresh:
        expert_list = []
        with open(llmPath, 'rb') as f:
            expert_list_dump = pickle.load(f)
        if len(expert_list_dump) != numOfExpert:
            print(f"in expertGenerator method: 读取本地{stage}专家缓存文件，但缓存文件数目与设定数目不一致，因此重新生成")
            return generateNewExpert(
                debug=Debug,
                Mode=Mode,
                llmType=llmType,
                numOfExpert=numOfExpert,
                stage=stage,
                llmPath=llmPath
            )
        for llm_msg in expert_list_dump:
            llm = init_raw_experts(debug=Debug, llmType=llmType, llmName=llm_msg['llm_name'])
            llm.load_data(llm_msg)
            expert_list.append(llm)
        print(f"in expertGenerator method: 读取本地{stage}专家缓存文件，共{len(expert_list_dump)}个专家")
    else:
        expert_list = generateNewExpert(
            debug=Debug,
            Mode=Mode,
            llmType=llmType,
            numOfExpert=numOfExpert,
            stage=stage,
            llmPath=llmPath
        )

    return expert_list


if __name__ == '__main__':
    web_llm = insert_knowledge(init_experts())