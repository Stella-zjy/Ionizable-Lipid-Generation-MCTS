import pandas as pd

df = pd.read_csv('chemprop_training_data.csv')

print(len(df))

print(len(df[df['target']==1]))
print(len(df[df['target']==0]))