"""This file contains the openAPI function."""

import time
import openai

#set openai key
openai.api_key = "" 


def chatgpt_request(inputs, model = "gpt-3.5-turbo", 
        system_role="You are a helpful assistant.", 
        num_retries=3, waiting_time = 1):
    """Obtain the response from GPT-3.5 API."""
    result = ''
    for _ in range(num_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                        {"role": "system", "content": system_role},
                        {"role": "user", "content": inputs},
                ]
            )
            for r in response.choices:
                result += r.message.content
            break
        except openai.error.OpenAIError as exception:
            print(f"{exception}. Retrying...")
            time.sleep(waiting_time)
    return result
