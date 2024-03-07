from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import os
import itertools
import json
import seaborn

api_key = os.environ["MISTRAL_API_KEY"]
client = MistralClient(api_key=api_key)

ad_beginning_phrases = [
    "Before we dive into today's episode, a quick word from our sponsor",
    "We're taking a short break to thank our sponsor",
    "A special thanks to our sponsor for making this episode possible",
    "Now, let's hear a message from our sponsor",
    "We're pausing the action for a moment to tell you about",
    "This podcast is brought to you by",
    "We're supported by",
    "A quick shoutout to our sponsor",
    "Let's take a moment to acknowledge our sponsor",
    "Before we continue, we'd like to express our gratitude to"
]
ad_ending_phrases = [
    "And now, back to our regularly scheduled programming.",
    "That's it for our sponsor message, let's dive back into the episode.",
    "And with that, we're back to our discussion.",
    "Thanks again to our sponsor, and now, let's continue with the show.",
    "And we're back, thanks for listening to that important message.",
    "That's all from our sponsor, now let's get back to the podcast.",
    "And now, let's return to our conversation.",
    "Thanks for your support, and now, back to the episode.",
    "And we're back, thanks for your patience.",
    "That's the end of our sponsor's message, let's jump back into the show."
]

begin_reference_embeddings = []
end_reference_embeddings = []
transcription_embeddings = {}

def get_text_embedding(input):
    embeddings_batch_response = client.embeddings(
          model="mistral-embed",
          input=input
      )
    return embeddings_batch_response.data[0].embedding

with open('YKS_344_transcript_4.json', 'r') as file:
    transcription = json.loads(file.read())

for sentence in ad_beginning_phrases:
    print(sentence)
    begin_reference_embeddings.append(np.array(get_text_embedding(sentence)))

for sentence in ad_ending_phrases:
    print(sentence)
    end_reference_embeddings.append(np.array(get_text_embedding(sentence)))

transcription = [line for line in transcription if len(line['transcription']) != 0] #TODO: this shouldn't be done here, should be odne when the transcription is created

for line in transcription:
    print(line["transcription"])
    transcription_embeddings[line["elapsed_time"]] = np.array(get_text_embedding(line["transcription"])) #zzz

for s, te in zip(transcription, transcription_embeddings):
    distances = []
    for re in begin_reference_embeddings:
        distances.append(euclidean_distances([re], [transcription_embeddings[te]]))
    if any(d < 0.63 for d in distances):
        print("START", s['transcription'], distances) #zzz
    distances = []
    for re in end_reference_embeddings:
        distances.append(euclidean_distances([re], [transcription_embeddings[te]]))
    if any(d < 0.63 for d in distances):
        print("END", s['transcription'], distances) #zzz