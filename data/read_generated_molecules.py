import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Data/6_generations_lipid/max_reactions_2_num_expand_nodes_1000/molecules.csv')

df_2_tails = df[df['num_reactions']==2]

print(len(df_2_tails))

reaction_dict = dict()
for row in df_2_tails.itertuples():
    if (int(row.reaction_1_id), int(row.reaction_2_id)) in reaction_dict:
        reaction_dict[(int(row.reaction_1_id), int(row.reaction_2_id))] += 1
    else:
        reaction_dict[(int(row.reaction_1_id), int(row.reaction_2_id))] = 1

print(sorted(reaction_dict.items(), key=lambda item: item[1], reverse=True))


# Plotting the distribution of the 'score' column using a histogram
plt.hist(df_2_tails['score'], bins=50, color='grey', alpha=0.7)  # bins parameter defines the number of intervals
plt.title('Score Distribution of Generated Two-tail Lipids')
plt.xlabel('Lipid Classifier Score')
plt.ylabel('Frequency')
plt.savefig('score_dist.png')

count = 0
for row in df_2_tails.itertuples():
    print(row.smiles)
    print(row.score)
    print(row.building_block_1_1_smiles)
    print(row.building_block_1_2_smiles)
    print(row.building_block_2_2_smiles)
    print('\n')
    count += 1
    if count == 10:
        break
