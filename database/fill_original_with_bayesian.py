import pandas as pd

# fromDF_r = "bayesian_criminal_filled_Bayesian.csv"
# toDF_r = "bayesian_criminal_filled.csv"
fromDF_r = "bayesian_victim_and_others_filled_Bayesian.csv"
toDF_r = "bayesian_victim_and_others_filled.csv"

fromDF = pd.read_csv(fromDF_r)

fromDF_dict = fromDF.to_dict()

toDF_dict = {}

for name in fromDF_dict:
    values = [fromDF_dict[name][i] for i in fromDF_dict[name]]
    valuesSet = set(values)
    valueMap = {
        value: index for index, value in enumerate(valuesSet)
    }
    toDF_dict[name] = []
    for index, value in enumerate(values):
        toDF_dict[name].append(valueMap[value])

toDF = pd.DataFrame(toDF_dict)
toDF.to_csv(toDF_r, index=False)
print()