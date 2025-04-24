import random
from datetime import datetime


class LargeLanguageModel:
    def __init__(self):
        self.base_url = ""
        self.api_key = ""
        self.default_llm_identity = "assistant"
        self.default_user_identity = "user"
        self.chat_history = []
        self.log_history = False
        self.log_pth = None
        self.debug_mode = False
        self.client = None
        self.model_name = None
        self.llm_name = None
        self.expertKnowledge = None

        adjectives = [
            "Brilliant", "Clever", "Fast", "Smart", "Wise", "Friendly", "Creative", "Reliable",
            "Efficient", "Inspiring", "Innovative", "Curious", "Ambitious", "Bold", "Calm",
            "Diligent", "Dynamic", "Empathetic", "Energetic", "Fantastic", "Fearless", "Genius",
            "Helpful", "Imaginative", "Ingenious", "Intelligent", "Kind", "Logical", "Loyal",
            "Optimistic", "Passionate", "Patient", "Powerful", "Proactive", "Productive",
            "Quick", "Resilient", "Resourceful", "Sharp", "Skillful", "Supportive", "Thoughtful",
            "Trustworthy", "Unstoppable", "Visionary", "Zealous"
        ]

        nouns = [
            "Assistant", "Helper", "Bot", "Guide", "Advisor", "Thinker", "Solver", "Explorer",
            "Creator", "Learner", "Instructor", "Partner", "Innovator", "Strategist", "Collaborator",
            "Teacher", "Visionary", "Friend", "Navigator", "Architect", "Builder", "Planner",
            "Researcher", "Mentor", "Analyst", "Consultant", "Pioneer", "Guardian", "Supporter",
            "Pathfinder", "Catalyst", "Coach", "Specialist", "Engineer", "Communicator",
            "Protector", "Facilitator", "Organizer", "Achiever", "Observer", "Inventor",
            "Dreamer", "Philosopher", "Mediator", "Advocate"
        ]

        # 随机选择形容词和名词生成名字
        random_adjective = random.choice(adjectives)
        random_noun = random.choice(nouns)
        # 加入时间戳
        self.llm_name = f"{random_adjective}-{random_noun}-{datetime.now().strftime('%Y%m%d')}"

    def refresh_2_baseKnowledge(self):
        self.chat_history = self.expertKnowledge
        self.save_logfile(info=self.generate_single_log(self.chat_history))

    def dump_data(self):
        # 将模型聊天记录等细节导出为一个数据结构
        outputData = {
            "llm_name": self.llm_name,
            "model_name": self.model_name,
            "chat_history": self.chat_history,
            "log_history": self.log_history,
            "log_pth": self.log_pth,
            "debug_mode": self.debug_mode,
            "expertKnowledge": self.expertKnowledge
        }
        return outputData

    def load_data(self, inputData):
        # 从数据结构中恢复模型聊天记录等细节，此时聊天记录只保存基本知识
        self.expertKnowledge = inputData['expertKnowledge']
        self.llm_name = inputData["llm_name"]
        self.model_name = inputData["model_name"]
        self.log_history = inputData["log_history"]
        self.log_pth = inputData["log_pth"]
        self.debug_mode = inputData["debug_mode"]

        # self.chat_history = inputData['expertKnowledge']
        self.refresh_2_baseKnowledge()

    def _init_llm(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def generate_single_log(self, info: str):
        time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info = str(info).replace('}, {', '}\n {')
        return 50*">" + f"[{time_stamp}]({self.llm_name}):\n {info}"


    def open_debug_mode(self):
        self.debug_mode = True

    def init_log_pth(self, log_pth):
        self.log_pth = log_pth
        with open(log_pth, "w") as f:
            f.write("")

    def save_logfile(self, info: str):
        if not self.log_history:
            return
        if self.log_pth is not None:
            with open(self.log_pth, "a") as f:
                f.write(self.generate_single_log(info) + "\n")

    def response(self, messages):
        # raise Exception("请重写本方法")
        """
                根据提供的消息生成聊天机器人的回复。

                此函数调用OpenAI的ChatCompletion API来生成回复消息。它首先使用类中定义的模型名称和提供的消息列表
                创建一个聊天完成对象，然后将该对象序列化为JSON格式的字符串并返回。这个过程涉及到与OpenAI API的网络通信，
                因此需要处理网络请求和响应。

                参数:
                messages_ (list): 一个消息字典的列表，每个字典包含角色（如"system", "user", "assistant"）和内容。
                                 例如: [{"role": "user", "content": "你好"}]

                返回:
                str: 一个包含聊天机器人回复信息的复合格式
                """
        # 创建聊天完成对象
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,  # 使用定义的模型名称
                messages=messages,  # 提供的消息列表
            )
        except Exception as e:
            raise Exception(f"回答信息时报错\n错误内容：{e}\n原文信息：{messages}")
        if self.log_history:
            self.chat_history.append(
                {"role": self.default_llm_identity, "content": completion.choices[0].message.content})
            if self.log_pth is not None:
                # self.save_logfile(info=self.generate_single_log(self.chat_history))
                self.save_logfile(
                    info={"role": self.default_llm_identity, "content": completion.choices[0].message.content})

        return completion

    def response_only_text(self, message):
        # raise Exception("请重写本方法")
        """
        根据输入的消息生成一个只包含文本的响应。

        此方法主要用于处理接收到的消息，并返回一个由OpenAI模型生成的，
        仅包含文本内容的响应。它会从模型的响应中提取出最相关的文本信息。

        参数:
        messages_ (list): 包含消息的列表，这些消息将被用来生成响应。

        返回:
        str: 由OpenAI模型生成的，与输入消息相关的文本内容。
        """
        response = self.response(message)
        text_response = response.choices[0].message.content
        # if self.log_history: # duplicated codes
        #     self.chat_history.append({"role": self.default_llm_identity, "content": text_response})
        return text_response

    def reset_history(self):
        self.chat_history = []

    def open_history_log(self):
        self.log_history = True

    def generate_msg(self, input_msg):
        if self.log_history:
            self.chat_history.append({"role": self.default_user_identity, "content": input_msg})
            if self.log_pth is not None:
                self.save_logfile(info={"role": self.default_user_identity, "content": input_msg})
                # self.save_logfile(info=self.generate_single_log(self.chat_history))
        return [{"role": self.default_user_identity, "content": input_msg}] if not self.log_history else self.chat_history

    def step_back(self):
        # 回退一次历史记录
        self.chat_history.pop()


if __name__ == '__main__':
    llm = LargeLanguageModel()
    print(llm.llm_name)