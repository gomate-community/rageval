from openai import OpenAI
client = OpenAI(
    base_url="http://project.gomall.ac.cn:30590/notebook/tensorboard/wangwenshan/1161/v1",
    api_key="sk-123456789",
)

completion = client.chat.completions.create(
  model="/home/gomall/models/Qwen2-7B-Instruct",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)