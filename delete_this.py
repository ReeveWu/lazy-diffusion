from utils.LLM import RetrievalWithPrompt, IsGenerationalRequest, GenerateDescriptions, GenerateStyle, StyleCommandDistinguisher

query = input("Enter your query: ")
style_command_distinguisher = StyleCommandDistinguisher()
print(style_command_distinguisher.invoke(query))