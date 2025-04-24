from Meta_LLM import LargeLanguageModel
from openai import OpenAI


"""
client = OpenAI(api_key="sk-a21249fcda9d4644aac224d9e017dc14", base_url="https://api.deepseek.com")
# Round 1
messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=messages
)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

print(reasoning_content)
print(content)
"""

class Deepseek_R1(LargeLanguageModel):
    def __init__(self, name=None):
        super().__init__()
        self.api_key = "sk-a21249fcda9d4644aac224d9e017dc14"
        self.default_llm_identity = 'system'
        if name is not None:
            self.llm_name = name
        self._init_llm(
            OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
            ),
            "deepseek-reasoner"
        )

    def CoT(self, msgLst: str):
        response = self.client.chat.completions.create(
            model="deepseek-reasoner",
            messages=msgLst
        )

        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        return reasoning_content, content

    def response_only_text(self, messageLst):
        reasoning_content, answer_content = self.CoT(messageLst)
        return answer_content

    def response_all_context(self, messages):
        reasoning_content, answer_content = self.CoT(messages)
        return reasoning_content, answer_content


if __name__ == '__main__':
    LLM = Deepseek_R1()
    print(LLM.response_all_context(LLM.generate_msg("你好，帮我查一下今天是农历几月几日")))
