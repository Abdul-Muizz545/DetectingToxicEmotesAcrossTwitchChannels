# Detecting toxic emotes across Twitch channels
This is an NLP project which extracts toxic emotes from HasanAbi's chat logs and uses a Word2Vec embedding space (trained on Twitch chat data and has emotes as well) to detect toxic emotes in other channels (for the purpose of testing we chose xQc and Pokimane's channel).
To compare emotes in the embedding space, cosine similarity was the metric we used. If a toxic HasanAbi emote is similar to an emote in another channel, then that emote is also classified as toxic.

This repository goes through the entire pipeline process of extracting toxic emotes from HasanAbi chat logs, visualizing the dataset, using t-SNE to identify other potential toxic HasanAbi emotes not discovered from the chat logs, and using those toxic and potentially toxic HasanAbi emotes to identify toxic emotes in other channels. The purpose of this is to help moderators since they are more aware to look out for messages that contain those toxic emotes. In addition, viewers of the Twitch stream will enjoy the experience more since there is less toxicity.

# Dataset used:
We used the Emotes-2-Vec dataset available at [](https://zenodo.org/records/8012284).  
