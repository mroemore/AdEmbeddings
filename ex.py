import requests as r
import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-small"
client = MistralClient(api_key=api_key)
system_prompt = "You are a helpful AI assistant. You will pay close attention to the level of detail provided in your answer. Your answers should be as short as possible unless the user specifies otherwise. The accuracy of your answers and the quality of your advice is of the utmost importance."

def get_da_LLM_output(user_prompt):
    msgs = []
    msgs.append(ChatMessage(role='system', content=system_prompt))
    msgs.append(ChatMessage(role='user', content=user_prompt))
    stream = client.chat(
        model=model, 
        messages=msgs
    )
    return stream.choices[0].message.content