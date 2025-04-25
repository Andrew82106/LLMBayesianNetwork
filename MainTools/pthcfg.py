import os
import sys


class PathConfig:
    def __init__(self):
        self.root_file_name = "114项目"
        self.cur_path = os.path.dirname(os.path.abspath(__file__))
        # cur_path中包含了root_file_name
        self.root_path = os.path.dirname(self.cur_path)
        self.main_tools_path = os.path.join(self.root_path, "MainTools")
        self.database_path = os.path.join(self.root_path, "database")

        self.assets_path = os.path.join(self.main_tools_path, "assets")
        self.llm_path = os.path.join(self.main_tools_path, "LLM")
        # 知识库json文件
        self.knowledge_base_json = os.path.join(self.assets_path, "KnowledgeBase.json")
        # 节点信息json文件
        self.nodes_info_json = os.path.join(self.assets_path, "nodes.json")

        # 日志文件夹
        self.log_pth = os.path.join(self.assets_path, "Log")
        # 缓存文件夹
        self.cache_path = os.path.join(self.assets_path, "cache")
        self.final_output_path = os.path.join(self.database_path, "output")

        sys.path.append(self.main_tools_path)
        sys.path.append(self.database_path)
        sys.path.append(self.assets_path)
        sys.path.append(self.llm_path)
        sys.path.append(self.log_pth)
        sys.path.append(self.cache_path)
        sys.path.append(self.log_pth)

        if not os.path.exists(self.log_pth):
            os.makedirs(self.log_pth)


if __name__ == '__main__':
    print(PathConfig().root_path)
    print(PathConfig().main_tools_path)
    print(PathConfig().database_path)

    print(PathConfig().assets_path)
    print(PathConfig().llm_path)

    print(PathConfig().log_pth)
    print(PathConfig().cache_path)
    print(PathConfig().final_output_path)

    print(PathConfig().knowledge_base_json)
    print(PathConfig().nodes_info_json)