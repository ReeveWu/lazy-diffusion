from utils.LLM import (
    GenerateDescriptions, 
    GenerateStyle, 
    RetrievalWithPrompt, 
    Conversation, 
    StyleCommandDistinguisher, 
    IsGenerationalRequest,
    UltimateRefiner
)
from utils.post_to_imgur import img_post
from utils.gen_img import txt2img

retriever = RetrievalWithPrompt(mode=1)
description_advisor = GenerateDescriptions()
style_advisor = GenerateStyle()
agent = Conversation()
style_distinguisher = StyleCommandDistinguisher()
finisher = IsGenerationalRequest()
ultimate_refiner = UltimateRefiner()

print("====================================")
print("All models are loaded.")
print("====================================")
print("Test the models: RetrievalWithPrompt")
print()
print(retriever.invoke("A beautiful sunset over the ocean."))
print("====================================")
print("Test the models: IsGenerationalRequest")
print()
print(finisher.invoke("A dog is playing with a toy"))
print("====================================")
print("Test the models: GenerateStyle")
print()
print(style_advisor.invoke("A dog is playing with a toy"))
print("====================================")
print("Test the models: GenerateDescriptions")
print()
print(description_advisor.invoke("A cat is playing with a stick."))
print("====================================")
print("Test the models: UltimateRefiner")
print()
print(ultimate_refiner.invoke("A dog is playing with a toy", ["abstract", "realism"], "A dog is playing with a toy in the park."))
print("====================================")