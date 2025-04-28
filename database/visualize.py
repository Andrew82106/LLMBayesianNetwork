import pandas as pd
import matplotlib.pyplot as plt
bayesian_criminal_filled = pd.read_csv("D:/Projects/LLMBayesianNetwork/database/bayesian_criminal_filled.csv")
bayesian_victim_and_others_filled = pd.read_csv("D:/Projects/LLMBayesianNetwork/database/bayesian_victim_and_others_filled.csv")

# 画出Loss列的直方图
bayesian_criminal_filled['Loss'].hist(bins=6)
plt.show()

bayesian_victim_and_others_filled['Loss'].hist(bins=6)
plt.show()


