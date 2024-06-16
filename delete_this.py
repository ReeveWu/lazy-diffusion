from utils.LLM import RetrievalWithPrompt, IsGenerationalRequest, GenerateDescriptions, GenerateStyle, StyleCommandDistinguisher

style_command_distinguisher = StyleCommandDistinguisher()
print(style_command_distinguisher.invoke("I want a preview of the style of Watercolor"))
print(style_command_distinguisher.invoke("I want to try out the style of anime"))
print(style_command_distinguisher.invoke("I want to see the final result"))
print(style_command_distinguisher.invoke("add the style of oil painting"))
print(style_command_distinguisher.invoke("anime"))
print(style_command_distinguisher.invoke("that's it"))
print(style_command_distinguisher.invoke("what it will lokk like in anime style"))
print(style_command_distinguisher.invoke("how it will look like in anime style"))