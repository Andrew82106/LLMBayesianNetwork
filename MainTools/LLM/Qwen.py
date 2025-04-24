from Meta_LLM import LargeLanguageModel
from openai import OpenAI


class Qwen(LargeLanguageModel):
    def __init__(self, name=None):
        super().__init__()
        self.api_key = "sk-9b0515f545954ca2bd65adfdc676e828"
        # self.client = OpenAI(
        #     api_key=self.api_key,
        #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        # )
        # self.model_name = "qwen-plus-0919"
        # self.model_name = "qwen-max-0919"
        self.default_llm_identity = 'system'
        if name is not None:
            self.llm_name = name
        self._init_llm(
            OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            "qwen-max-0919"
        )


class QWQ(LargeLanguageModel):
    def __init__(self, name=None):
        super().__init__()
        self.api_key = "sk-9b0515f545954ca2bd65adfdc676e828"
        self._init_llm(
            OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            "qwq-32b"
        )

    def CoT(self, msgLst: str):
        reasoning_content = ""  # 定义完整思考过程
        answer_content = ""  # 定义完整回复
        is_answering = False  # 判断是否结束思考过程并开始回复
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=msgLst,
            # messages=[
            #     {"role": "user", "content": msgIn}
            # ],
            # QwQ 模型仅支持流式输出方式调用
            stream=True,
            # 解除以下注释会在最后一个chunk返回Token使用量
            # stream_options={
            #     "include_usage": True
            # }
        )
        reasoningContent = ""
        answerContent = ""
        for chunk in completion:
            # 如果chunk.choices为空，则打印usage
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                # 打印思考过程
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    # print(delta.reasoning_content, end='', flush=True)
                    reasoning_content += delta.reasoning_content
                    reasoningContent += delta.reasoning_content
                else:
                    # 开始回复
                    if delta.content != "" and is_answering is False:
                        # print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                        is_answering = True
                    # 打印回复过程
                    # print(delta.content, end='', flush=True)
                    answer_content += delta.content
                    answerContent += delta.content

        return reasoningContent, answerContent

    def response_only_text(self, messageLst):
        reasoning_content, answer_content = self.CoT(messageLst)
        return answer_content

    def response_all_context(self, messages):
        reasoning_content, answer_content = self.CoT(messages)
        return reasoning_content, answer_content


class QWQ_plus(LargeLanguageModel):
    def __init__(self, name=None):
        super().__init__()
        self.api_key = "sk-9b0515f545954ca2bd65adfdc676e828"
        self._init_llm(
            OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            "qwq-plus"
        )

    def CoT(self, msgLst: str):
        reasoning_content = ""  # 定义完整思考过程
        answer_content = ""  # 定义完整回复
        is_answering = False  # 判断是否结束思考过程并开始回复
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=msgLst,
            # messages=[
            #     {"role": "user", "content": msgIn}
            # ],
            # QwQ 模型仅支持流式输出方式调用
            stream=True,
            # 解除以下注释会在最后一个chunk返回Token使用量
            # stream_options={
            #     "include_usage": True
            # }
        )
        reasoningContent = ""
        answerContent = ""
        for chunk in completion:
            # 如果chunk.choices为空，则打印usage
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                # 打印思考过程
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    # print(delta.reasoning_content, end='', flush=True)
                    reasoning_content += delta.reasoning_content
                    reasoningContent += delta.reasoning_content
                else:
                    # 开始回复
                    if delta.content != "" and is_answering is False:
                        # print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                        is_answering = True
                    # 打印回复过程
                    # print(delta.content, end='', flush=True)
                    answer_content += delta.content
                    answerContent += delta.content

        return reasoningContent, answerContent

    def response_only_text(self, messageLst):
        reasoning_content, answer_content = self.CoT(messageLst)
        return answer_content

    def response_all_context(self, messages):
        reasoning_content, answer_content = self.CoT(messages)
        return reasoning_content, answer_content


if __name__ == '__main__':
    LLM = QWQ_plus()
    print(LLM.response_all_context(LLM.generate_msg("你好，帮我查一下今天是农历几月几日")))
