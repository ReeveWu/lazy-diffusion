from utils.LLM import RetrievalWithPrompt, IsGenerationalRequest, GenerateDescriptions, GenerateStyle, StyleCommandDistinguisher, ultimate_refiner

description_advisor = GenerateDescriptions()
ans1 = description_advisor.invoke("a cat is running on the street")["result"]
ans2 = description_advisor.invoke("A lion sitting on the chair")["result"]
ans3 = description_advisor.invoke("an aisin woman cooking")["result"]

print(len(ans1))
print(len(ans2))
print(len(ans3))


print(ans1)
print(ans2)
print(ans3[2])
