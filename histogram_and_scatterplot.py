"""
This file has code to plot the histograms for the toxicity ratio of emotes we extracted
In addition, we made scatter plots for the number of times emote was used vs toxicity ratio.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.special import kl_div, rel_entr
from scipy.spatial.distance import jensenshannon


#Read csv file
df = pd.read_csv("Emote_analysis.csv")

#Count nulls in each column (there are no nulls)
print(df.isna().sum()) 
print(df.dtypes)



#Retrieving toxicity ratio for all emotes, global emotes and channel emotes
df_toxicityratio = df["Toxicity Ratio"]
df_toxicityratioglobalemotes = df[df["Global emote"] == True]["Toxicity Ratio"].reset_index(drop = True)
df_toxicityratiochannelemotes = df[df["Channel emote"] != "False"]["Toxicity Ratio"].reset_index(drop = True)


#Retrieving emote frequency for all emotes, global emotes, and channel emotes
df_emotefrequency = df["Number of times emote was toxic"] + df["Number of times emote was NOT toxic"]
df["Emote frequency"] = df_emotefrequency
df_emotefrequencyglobalemotes = df[df["Global emote"] == True]["Emote frequency"].reset_index(drop = True)
df_emotefrequencychannelemotes = df[df["Channel emote"] !="False"]["Emote frequency"].reset_index(drop = True)


###PLOT 1a : HISTOGRAM FOR TOXICITY RATIO OF ALL EMOTES####################

# Plot the histogram for toxicity ratio of all emotes
heights1, bin_edges1, patches1 = plt.hist(df_toxicityratio, bins = int((df_toxicityratio.max() - df_toxicityratio.min()) / 0.2), color = "blue")
plt.title('Histogram of toxicity ratio (all emotes)')
plt.xlabel('Toxicity Ratio')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#Dividing histogram bar heights by sum to turn it into a probability mass function
normalized_heights1 = heights1 / sum(heights1)


###PLOT 1b: HISTOGRAM FOR TOXICITY RATIO OF only global emotes#############
heights2, bin_edges2, patches2  = plt.hist(df_toxicityratioglobalemotes, bins = int((df_toxicityratioglobalemotes.max() - df_toxicityratioglobalemotes.min()) / 0.2), color = "blue")
plt.title('Histogram of toxicity ratio (global emotes)')
plt.xlabel('Toxicity Ratio')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

normalized_heights2 = heights2/ sum(heights2)

###PLOT 1c: HISTOGRAM FOR TOXICITY RATIO OF only channel emotes#############
heights3, bin_edges3, patches3  = plt.hist(df_toxicityratiochannelemotes, bins = int((df_toxicityratiochannelemotes.max() - df_toxicityratiochannelemotes.min()) / 0.2), color = "blue")
plt.title('Histogram of toxicity ratio (channel emotes)')
plt.xlabel('Toxicity Ratio')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

normalized_heights3 = heights3/ sum(heights3)

########## PLOT 2a: SCATTER PLOT OF (Emote frequency) vs  (toxicity ratio) for all emotes#############
plt.scatter(df_toxicityratio, df_emotefrequency)
plt.yscale('log')
plt.title('Scatter plot of emote frequency vs toxicity ratio for all emotes')
plt.xlabel('Toxicity Ratio')
plt.ylabel('Emote Frequency')
plt.show()

########## PLOT 2b: SCATTER PLOT OF (Emote frequency) vs (toxicity ratio) for only global emotes #############
plt.scatter(df_toxicityratioglobalemotes, df_emotefrequencyglobalemotes)
plt.yscale('log')
plt.title('Scatter plot of emote frequency vs toxicity ratio for global emotes')
plt.xlabel('Toxicity Ratio')
plt.ylabel('Emote Frequency')
plt.show()

########## PLOT 2c: SCATTER PLOT OF (Emote frequency) vs (toxicity ratio) for only channel emotes  #############
plt.scatter(df_toxicityratiochannelemotes, df_emotefrequencychannelemotes)
plt.yscale('log')
plt.title('Scatter plot of emote frequency vs toxicity ratio for channel emotes')
plt.xlabel('Toxicity Ratio')
plt.ylabel('Emote Frequency')
plt.show()

########### COMPUTING KL-DIVERGENCE between histograms to see how much they differ from each other ##########
#KL-Divergence is not symmetrical meaning that KL_div(a, b) not equal to KL_div(b, a)
print(sum(kl_div(normalized_heights2,  normalized_heights3)))
print(sum(rel_entr(normalized_heights2, normalized_heights3)))

print(sum(kl_div(normalized_heights3,  normalized_heights2)))
print(sum(rel_entr(normalized_heights3, normalized_heights2)))


########### COMPUTING JS-DISTANCE between histograms to see how much they differ from each other ##########

#JS-Divergence is symmetrical and uses KL-divergence in the formula. it is finite and the returned value is a DISTANCE
#JS-divergence provides a smoothed and normalized version of KL divergence, with scores between 0 (identical) and 1 (maximally different), when using the base-2 logarithm.

js_distance = jensenshannon(normalized_heights2, normalized_heights3, base=2)
print("JS distance between histogram for global and channel emotes is: %.2f " % js_distance)

#JS distance between histogram for global and channel emotes we got is 0.03. This is very small which tells us that the distributions are veery close to each other
