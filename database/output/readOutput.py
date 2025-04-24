import pickle as pkl


def readOutput(filename):
    with open(filename, 'rb') as f:
        content = pkl.load(f)
    return content


result = readOutput('ExpertResult_labData[time=2025-04-2420_35_42].pkl')
print(result['result']['Graph']['Victimhuman'][1].edges())
print(result['result']['Graph']['Criminalhuman'][1].edges())
print()