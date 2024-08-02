# Detecting toxic emotes across Twitch channels
This is an NLP project which extracts toxic emotes from HasanAbi's chat logs and uses a Word2Vec embedding space to detect toxic emotes in other channels (for the purpose of testing we chose xQc and Pokimane's channel).
To compare emotes in the embedding space, cosine similarity was the metric we used. If a toxic HasanAbi emote is similar to an emote in another channel, then that emote is also classified as toxic.

This repository goes through the entire pipeline process of extracting toxic emotes from HasanAbi chat logs, visualizing the dataset, using t-SNE to identify other potential toxic HasanAbi emotes not discovered from the chat logs, and using those toxic and potentially toxic HasanAbi emotes to identify toxic emotes in other channels. The purpose of this is to help moderators so that they are more aware to look out for messages that contain those toxic emotes. In addition, viewers of the Twitch stream will also enjoy the experience of chatting a lot more since there is less toxicity.

# Dataset used:
I used the public dataset available at [Emotes-2-Vec](https://zenodo.org/records/8012284). This dataset was created by downloading Twitch chat data from over 2000 Twitch channels, cleaning the data and then combining it into one large corpus called *corpus_raw.txt*. From there, they trained a Word2Vec model on that corpus and created the Word2Vec embedding space that we used for our project. While HasanAbi's chat data that has been cleaned is part of the corpus, it is almost impossible to find in that large corpus without any information that indicates which channel the comment was sent in. Hence, with the permission of the author of Emotes-2-Vec, we borrowed HasanAbi's raw chat logs, removed the username and timestamp columns and uploaded them to the folder HasanAbiData.

# How to run the code:
1. I did not put the exact file paths when reading HasanAbi's chat logs from the HasanAbi data folder, so you would need to change those filepaths to wherever you stored the chat logs.
2. The Emotes-2-Vec dataset was downloaded into a folder called Data on my local machine. Therefore, whenever you see "Data\filename" in the filepath, replace it to wherever you stored the public dataset followed by the filename I used in the file path.
3. The order of how you should run the code is indicated by the number before the filename. For example, 1_robertaclassifer.py is the first step of the pipeline.
