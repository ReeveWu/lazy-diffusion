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
        self.retriever = self.db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5, "k": 5})

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
You are an assistant responsible for determining whether a user's input is a description of a scenario or an image. \
If not, output 'no'; if it is, output 'yes'. \
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
You are an AI assistant specialized in refining scene descriptions for image generation purposes. \
Your task is to transform a basic description provided by the user into three unique, vivid, and artistically enriched scene variations. \
Focus on infusing each description with creative elements that enhance visual appeal and artistic quality, ensuring they remain short and impactful. \
Each variation must be concise, detailed, and not exceed 10 tokens in length. \

Your response should adhere to the following format: a JSON object with a single key 'result', and the value should be a list containing three distinct scene descriptions, And no preamble or explanation.
This is an example of the expected output: {{"result": ["<description1>", "<description2>", "<description3>"]}} \
<|eot_id|><|start_header_id|>user<|end_header_id|>\
Description: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["query"]
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


class Conversation(LLM):
    def __init__(self):
        self.llm = Ollama(model="llama3")
        self.chain = self.get_prompt() | self.llm | StrOutputParser()

    def invoke(self, query):
        return self.chain.invoke({"query": query})

    @staticmethod
    def get_prompt():
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> \
You are an agent who can help users to use the tool that can optimize the prompts they use for generating images. Users can start using the service by typing "Get started" on the chatbox. Please chat with the users.
Your response must be simple and clear. \
Output within 10 words. \
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
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> \
You are a specialized assistant tasked with interpreting user inputs related to image styling tasks. \
Determine if the input is a request to add a style to an image, preview a style, or generate the final styled image. \
Classify the input into one of these actions: 'add', 'preview', or 'final'.
If the user is wondering what the style will look like, the action should be 'preview'.
The defualt action is 'add', if the user's input imply wanting to end the styling process, the action should be 'final'.
If the user want a preview, sample or example of the style, the action should be 'preview'.

If the action is 'add' or 'preview', extract the style name from the input. If the action is 'final', the style name should be an empty string.

Your response should be structured as a JSON object with two keys: 'mode' and 'style'. \
Answer it without any description and no preamble or explanation.\
The 'mode' key should reflect the user's intent, being either 'add', 'preview', or 'final'. \
The 'style' key should contain the style name if applicable, and be an empty string if the mode is 'final'.
<|eot_id|><|start_header_id|>user<|end_header_id|>\
{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["query"],
        )
    


#create a comprehensive final prompt using previous results
class UltimateRefiner(LLM):
    def __init__(self):
        super().__init__(self.get_prompt())

    def invoke(self, base_prompt, style, description):
        return self.chain.invoke({"base_prompt": base_prompt, "style": style, "description": description})

    @staticmethod
    def get_prompt():
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> \
You are an AI trained to merge three key inputs: \
a 'base_prompt' describing the core idea, a 'style' indicating the artistic or stylistic approach, \
and a 'description' providing additional details. \
Your task is to blend these inputs into a coherent, aesthetically appealing prompt suitable for generating images with Stable Diffusion.
Your responses should consist of a single, well-integrated prompt without any preamble or explanation.
user

your response should be less than 30 words.
Your answer should be a JSON with a single key 'result' and a your response and no preamble or explanation. \
<|eot_id|><|start_header_id|>user<|end_header_id|>\
Base Prompt: {base_prompt}, Style: {style}, Description: {description}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["base_prompt", "style", "description"],
        )