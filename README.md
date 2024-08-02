# Detecting toxic emotes across Twitch channels
This is an NLP project which extracts toxic emotes from HasanAbi's chat logs, and then uses a Word2Vec embedding space to detect toxic emotes in other channels (for the purpose of testing we chose xQc and Pokimane's channel).
To compare emotes in the embedding space, cosine similarity was the metric we used. If a toxic HasanAbi emote is similar to an emote in another channel, then that emote is also classified as toxic.
