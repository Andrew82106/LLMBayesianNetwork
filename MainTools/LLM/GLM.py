from zhipuai import ZhipuAI
import requests
from openai import OpenAI
from Meta_LLM import LargeLanguageModel



class ChatGLM_Origin_Zhipu(LargeLanguageModel):
    def __init__(self, name=None):
        super().__init__()
        self.api_key = "33b333df733a7ba7174034ef5d757c8f.1MlCkHLb22BysIPi"
        self.default_llm_identity = 'system'
        if name is not None:
            self.llm_name = name
        self._init_llm(
            ZhipuAI(api_key=self.api_key),
            "glm-4-flash"
        )
        # self.client = ZhipuAI(api_key=self.api_key)  # 请填写您自己的APIKey


if __name__ == '__main__':
    LLM = ChatGLM_Origin_Zhipu()
    messages = LLM.generate_msg("你好，李白是谁")
    print(LLM.response_only_text(messages))