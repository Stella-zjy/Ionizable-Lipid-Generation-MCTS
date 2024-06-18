import pandas as pd
import matplotlib.pyplot as plt
import os


# ====================== Concatenate all heads ======================
# Specify the directory to search
# directory = 'Data/RawLipid/Data01'

# # List all files in the directory
# files_in_directory = os.listdir(directory)

# # Filter out all files that end with '.csv'
# head_file_list = [file for file in files_in_directory if file.endswith('.csv')]

# dataframes = []

# # Loop through the files and read them into separate dataframes
# for filename in head_file_list:
#     df = pd.read_csv(directory + '/' + filename)
#     print(filename,'size:', len(df), '\n')
#     dataframes.append(df)

# # Concatenate all the dataframes into one
# concatenated_df = pd.concat(dataframes, ignore_index=True)
# print(len(concatenated_df))

# concatenated_df.to_csv('Data/RawLipid/heads.csv', index=False)

# print(concatenated_df.head())
# print(concatenated_df.tail())

# ====================== Create Building Blocks ======================
# head_df = pd.read_csv('Data/RawLipid/heads.csv')
# tail_df = pd.read_csv('Data/RawLipid/purchasable_tails.csv')

# print(len(head_df))
# print(len(tail_df))
# concatenated_df = pd.concat([head_df, tail_df], ignore_index=True)
# print(len(concatenated_df))

# # Adding a unique ID column
# concatenated_df['ID'] = range(1, len(concatenated_df) + 1)

# print(concatenated_df.head())
# print(concatenated_df.tail())

# # Writing to a new CSV
# concatenated_df.to_csv('Data/RawLipid/lipid_building_blocks.csv', index=False)



# ====================== Read Building Block Scores ======================
all_df = pd.read_csv('Data/RawLipid/lipid_building_blocks_with_preds.csv')

plt.figure()
head_df = all_df.head(2752482)
print(len(head_df))
plt.hist(head_df['chemprop_morgan_model_0_preds'], bins=100, color='grey', alpha=0.7)  # bins parameter defines the number of intervals
plt.title('Distribution of Head Lipid Classifier Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.savefig('head_score_dist.png')

plt.figure()
tail_df = all_df.tail(7412)
print(len(tail_df))
plt.hist(tail_df['chemprop_morgan_model_0_preds'], bins=100, color='grey', alpha=0.7)  # bins parameter defines the number of intervals
plt.title('Distribution of Tail Lipid Classifier Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.savefig('tail_score_dist.png')