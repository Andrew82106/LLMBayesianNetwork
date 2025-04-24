import pickle as pkl

def readOutput(filename):
    with open(filename, 'rb') as f:
        content = pkl.load(f)
    return content


result = readOutput('D:\Projects\LLMBayesianNetwork\database\output\labData_time=2025-04-22_18-04-22.pkl')
print()