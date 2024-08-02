"""
This file has code that removes short comments and uses the Roberta Toxicity Classifier to classify them as toxic or not toxic.
Our results are then written to a new csv file.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import pandas as pd

#read HasanAbi csv file (only need comment column)
df = pd.read_csv("HasanAbiData\ name of csv file you want to read", usecols = ["comment"])
#Note that you can read multiple HasanAbi csv files into seperate dataframes and then concatenate them into one.

df = df.dropna() #remove rows with null values

# Custom function to count number of words
def count_words(text):
    return len(text.split())


df['word_count'] = df['comment'].apply(count_words)

# Filter rows where word count is greater than 7
df = df[df['word_count'] > 7]
df = df.drop(columns=['word_count']).reset_index(drop = True)
print(df)
print("df now only contains comments with more than 7 words")

#Initializing tokenizer with pre-trained weights of roberta-toxicity-classifier model. Tokenizer will conver raw input into useful input for model
tokenizer = AutoTokenizer.from_pretrained("s-nlp/roberta_toxicity_classifier") 

#initializes a pre-trained model for sequence classification using the weights from "s-nlp/roberta_toxicity_classifier". The model is a RoBERTa-based model fine-tuned for toxicity classification.
model = AutoModelForSequenceClassification.from_pretrained("s-nlp/roberta_toxicity_classifier")


threshold = 0.5

counter = 0
for index, row in df.iterrows():    
    batch = tokenizer(row["comment"], return_tensors = "pt") #process the input text . 
    output = model(**batch) #obtain predictions from tensors made by tokenizer
    #print(output) #logits are unnormalized predictions from model

    logits = output.logits
    # softmax function to convert logits to probabilities
    probabilities = softmax(logits, dim=1)

    #probability for the positive class (toxic)
    toxic_probability = probabilities[0, 1].item()
    
    # Determine the toxicity based on a threshold
    is_toxic = toxic_probability > threshold

    # Update the "toxic" column for the current row
    df.at[index, "toxic"] = "yes" if is_toxic else "no"

    if counter % 1000 == 0:
        print(str(counter) + " rows have been processed")    

    # print(f"Probability of toxicity: {toxic_probability}")
    # print(f"The text is {'toxic' if is_toxic else 'non-toxic'}")
    counter += 1

print(df["toxic"].value_counts())
df.to_csv("output_toxicity.csv", index = False)

