from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate

from utils.Embeddings import EmbeddingModel


class LLM:
    def __init__(self, prompt):
        self.llm = Ollama(model="llama3")
        self.chain = prompt | self.llm | JsonOutputParser()

    def invoke(self, query):
        return self.chain.invoke({"query": query})


# Retrieve some reference prompt that is great for being the input of stable diffusion model
class RetrievalWithPrompt:
    def __init__(self, mode=1):
        self.llm = Ollama(model="llama3")
        self.embeddings = EmbeddingModel(mode)
        self.db = FAISS.load_local(
            "./utils/prompt_index",
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.retriever = self.db.as_retriever()

    def invoke(self, prompt):
        docs = self.retriever.invoke(prompt)
        return docs


# The model is used to determine whether the user's input is a request to start generating an image
class IsGenerationalRequest(LLM):
    def __init__(self):
        super().__init__(self.get_prompt())

    def invoke(self, query):
        return super().invoke(query)

    @staticmethod
    def get_prompt():
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> \
You are an assistant who helps users generate images. \
Your job is to determine whether the user's input is a request for you to start generating an image. \
Users might not explicitly make this request, so here is a tip: if the user's response is not describing the requirements for an image (including composition, style, etc.), \
and it's not part of a casual conversation, but instead has a tone of requesting action or indicates that they have finished explaining their requirements, \
this usually means they likely want you to start generating the image.

If it is, return 'yes'; otherwise, return 'no'.
Output the answer as a JSON with a single key 'result' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>\
{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["query"],
        )


# The model is used to generate three more complete and detailed scene descriptions based on the user's brief description
class GenerateDescriptions(LLM):
    def __init__(self):
        super().__init__(self.get_prompt())

    def invoke(self, query):
        return super().invoke(query)

    @staticmethod
    def get_prompt():
        return PromptTemplate(
                template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> \
You are a specialized assistant tasked with enhancing user-provided scene descriptions for image generation. \
Your goal is to enrich these brief descriptions into three distinct, detailed, and visually appealing scene variations. \
Focus on adding creative elements and details that increase the vividness and artistic quality of each scene.
Each description must within 10 words. \

Your answer should be a JSON with a single key 'result' and a list of three scene descriptions. and no preamble or explanation. \
Please double check that you're following the right format and structure and the response is NOT empty. \

<|eot_id|><|start_header_id|>user<|end_header_id|>\
{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["query"],
    )


# The model is used to generate six different styles based on the user's sentence
class GenerateStyle(LLM):
    def __init__(self):
        super().__init__(self.get_prompt())

    def invoke(self, query):
        return super().invoke(query)

    @staticmethod
    def get_prompt():
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> \
You are an artist whose job is to provide six different styles based on user sentences, \
capturing the context of the sentences to allow users to choose their preferred style. \
Your styles should be as diverse as possible, each represented by a single keyword.

Your answer should be a JSON with a single key 'result' and a list of six different styles without any description. and no preamble or explanation. \
<|eot_id|><|start_header_id|>user<|end_header_id|>\
{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["query"],
        )

# The model is used to distinguish whether the user's input is a request to add a style or preview a style or generate the final image
class StyleCommandDistinguisher(LLM):
    '''
    output styles:
    generate final image: "final"
    preview a style: "style"
    add a style: "<style_name>"
    '''
    def __init__(self):
        super().__init__(self.get_prompt())

    def invoke(self, query):
        return super().invoke(query)

    @staticmethod
    def get_prompt():
        return PromptTemplate(
#             template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> \
# You are a specialized assistant designed to interpret user inputs regarding image processing tasks. \
# Your task is to accurately determine whether the user's input is a request to add a style to an image, preview a style on an image, or generate the final styled image. \
# Based on the user's command, you should categorize the input into one of these three specific actions.


# Your response should be formatted as a JSON object with a two key 'mode' and 'style'. \
# The value of 'mode' should accurately reflect the user's intent based on their input and follow the specified format: "add", "preview", or "final". \
# The value of 'style' should be the name of the style if the user's input is a request to add a style or preview a style, if the user choose final, the value should be "" \
# <|eot_id|><|start_header_id|>user<|end_header_id|>\
# {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> \
You are a specialized assistant tasked with interpreting user inputs related to image styling tasks. \
Determine if the input is a request to add a style to an image, preview a style, or generate the final styled image. \
Classify the input into one of these actions: 'add', 'preview', or 'final'.

If the action is 'add' or 'preview', extract the style name from the input. If the action is 'final', the style name should be an empty string.

Your response should be structured as a JSON object with two keys: 'mode' and 'style'. \
Answer it without any description and no preamble or explanation.\
The 'mode' key should reflect the user's intent, being either 'add', 'preview', or 'final'. \
The 'style' key should contain the style name if applicable, and be an empty string if the mode is 'final'.
<|eot_id|><|start_header_id|>user<|end_header_id|>\
{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["query"],
        )
    


# create a comprehensive final prompt using previous results
# class ultimate_refiner(LLM):
#     def __init__(self):
#         super().__init__(self.get_prompt())

#     def invoke(self, base_prompt, style, description):
#         return super().invoke(query)

#     @staticmethod
#     def get_prompt():
#         return PromptTemplate(
#             template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> \
# You are an artist whose job is to provide six different styles based on user sentences, \
# capturing the context of the sentences to allow users to choose their preferred style. \
# Your styles should be as diverse as possible, each represented by a single keyword.

# Your answer should be a JSON with a single key 'result' and a list of six different styles without any description. and no preamble or explanation. \
# <|eot_id|><|start_header_id|>user<|end_header_id|>\
# {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
#             input_variables=["query"],
#         )