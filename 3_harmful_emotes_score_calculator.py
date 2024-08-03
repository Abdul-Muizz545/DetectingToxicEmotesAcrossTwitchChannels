"""
This file has code to determine which emotes are harmful and compute a harmful score for those emotes
We only consider emotes with toxicity ratio >=0.5 as being potential candidates of being toxic and harmful because they were used in a toxic conversation more than non toxic
where toxicity ratio for an emote = Number of times emote was toxic / (number of times emote was toxic + non toxic).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Read csv file for emotes extracted 
df = pd.read_csv("Emote_analysis.csv")

#Count nulls in each column (there are no nulls)
print(df.isna().sum()) 
print(df.dtypes)

#All emotes with a toxicity ratio < 0.5 dont have potential to be harmful so we will filter them out
df = df[df["Toxicity Ratio"] >= 0.5]

#Retrieving toxicity ratio and emote frequency
df_toxicityratio = df["Toxicity Ratio"]
df_emotefrequency = df["Number of times emote was toxic"] + df["Number of times emote was NOT toxic"]


#Defining weights for importance of emote frequency and toxicity ratio respectively in determining each emote's harmful score
weight_emotefreq = 1
weight_toxicityratio = 1

#Computing harmful score for the toxic emotes and writing them to a new output csv file
df["Harmful score"] = (weight_emotefreq * df_emotefrequency) + (weight_toxicityratio * df_toxicityratio)
df.reset_index(drop = True, inplace = True)
df.to_csv("Harmfulemotes_analysis.csv")
