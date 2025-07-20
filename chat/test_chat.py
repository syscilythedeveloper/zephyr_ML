from chat_engine import ZephyrChatEngine

# Initialize the chat engine
chat_engine = ZephyrChatEngine()

# Ask a question
question = input("Enter a question about the sentiment data: ")
response = chat_engine.chat(question)

# Print the answer
print(response["answer"])