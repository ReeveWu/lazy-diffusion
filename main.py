from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent, TextMessage,
    TextSendMessage, ImageSendMessage, TemplateSendMessage, FlexSendMessage,
    QuickReply, QuickReplyButton,
    ButtonsTemplate, MessageAction, URITemplateAction, URIAction, CarouselTemplate, CarouselColumn
)
from dotenv import load_dotenv
import os
import cv2

from linebot.models import MessageAction, TemplateSendMessage, ConfirmTemplate

from collections import defaultdict

from utils.LLM import GenerateDescriptions, GenerateStyle, RetrievalWithPrompt, Conversation, StyleCommandDistinguisher, IsGenerationalRequest
from utils.post_to_imgur import img_post
from utils.gen_img import txt2img

retriever = RetrievalWithPrompt(mode=1)
description_advisor = GenerateDescriptions()
style_advisor = GenerateStyle()
agent = Conversation()
style_distinguisher = StyleCommandDistinguisher()
finisher = IsGenerationalRequest()

load_dotenv()

CHANNEL_ACCESS_TOKEN = os.getenv('CHANNEL_ACCESS_TOKEN')
CHANNEL_SECRET = os.getenv('CHANNEL_SECRET')

def init_info():
    return {
        "state": -1,
        "preview_style": "",
        "rag": [],
        "description_suggestion": [],
        "suggested_styles": [], 
        "base_prompt": "",
        "description": "",
        "styles": [],
        "past_prompts": [],
    }

history = defaultdict(init_info)

# state 0: initial state
# state 1: user has sent a base prompt
# state 2: user wants to see prompt examples
# state 3: user are adding more details to the prompt
# state 4: user are selecting styles
# state 5: user has finished adding details based on the rag prompt
# state 6: user has finished selecting styles and is ready to generate the image
# state 7: user want to share the prompt or not
# state 8: user want to preview the image or not
# state -1: user has not started a round

app = Flask(__name__)

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)


@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    handler.handle(body, signature)
    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_message = event.message.text
    user_id = event.source.user_id
    print(event)
    user_state = history[user_id]["state"]
    
    if user_state == -1:
        if user_message == "Let's go!":
            messages = [
                TextMessage(text="Great! Let's start with a base prompt. "), 
                TextMessage(text="Please type a sentence that describes the image you want to generate."),
                TextMessage(text="p.s. If you have not decided yet, you can click 'Not now!' to exit.")
            ]
            history[user_id]["state"] = 0
            
        elif user_message == "Not now!":
            messages = [
                TextMessage(text="Okay, let me know when you're ready!"),
            ]
            
        elif user_message.lower() == "get started":
            messages = [
                TextMessage(text="Hi, I'm your image generation assistant. "),
                TextMessage(text="I will guide you step by step to improve your prompt for generating high-quality images."),
                TextMessage(text="Click the button below to embark on this magical journey!"),
                TemplateSendMessage(
                    alt_text='Get started',
                    template=ConfirmTemplate(
                            text="Are you ready?",
                            actions=[
                                MessageAction(
                                    label='Let\'s go!',
                                    text='Let\'s go!'
                                ),
                                MessageAction(
                                    label='Not now!',
                                    text='Not now!'
                                )
                            ]
                        )
                )
            ]
        else:
            llm_response = agent.invoke(user_message)
            messages = [
                TextMessage(text=llm_response)
            ]
        line_bot_api.reply_message(event.reply_token, messages)
    elif user_state == 0:
        if user_message == "Not now!":
            messages = [
                TextMessage(text="Okay, let me know when you're ready!")
            ]
            history[user_id]["state"] = 2
            line_bot_api.reply_message(event.reply_token, messages)
        else:
            history[user_id]['base_prompt'] = user_message
            messages = [
                TextMessage(text="Great! Acorrding to your query, I have found prompts that might be helpful for you. Do you want to see them?"),
                TemplateSendMessage(
                    alt_text='Prompt examples',
                    template=ConfirmTemplate(
                        text="Do you want to see prompt examples?",
                        actions=[
                            MessageAction(
                                label='Yes',
                                text='Yes'
                            ),
                            MessageAction(
                                label='No',
                                text='No'
                            )
                        ]
                    )
                )
            ]
            history[user_id]["state"] = 1
            line_bot_api.reply_message(event.reply_token, messages)
            
    elif user_state == 1:
        if user_message == "Yes":
            docs = retriever.invoke(history[user_id]['base_prompt'])
            for doc in docs:
                history[user_id]['rag'].append(doc.page_content)
            print("docs", docs)
            messages = [
                TextMessage(text="Here are some prompts that might be helpful for you: "),
                TemplateSendMessage(
                    alt_text='Suggestions',
                    template=CarouselTemplate(columns=[
                                CarouselColumn(
                                    text=doc.page_content[:57] + "..." if len(doc.page_content) > 60 else doc.page_content,
                                    thumbnail_image_url=doc.metadata['url'],
                                    actions=[
                                        MessageAction(label='I want this!', text=f"<RAGPROMPT {i}>"),
                                        URIAction(label='Show image', uri=doc.metadata['url']),
                                    ]
                                ) for i, doc in enumerate(docs)
                            ])
                ),
                TextSendMessage(
                    text='Or you can click the button below to continue.',
                    quick_reply=QuickReply(
                        items=[
                            QuickReplyButton(
                                action=MessageAction(label="Continue", text="Continue")
                            )]
                    ))
            ]
            history[user_id]["state"] = 2
            line_bot_api.reply_message(event.reply_token, messages)
            
        elif user_message == "No":
            try:
                suggestions = description_advisor.invoke(history[user_id]['base_prompt'])['result']
                if not isinstance(suggestions[0], str):
                    suggestions = []
            except:
                try:
                    suggestions = description_advisor.invoke(history[user_id]['base_prompt'])['result']
                    if not isinstance(suggestions[0], str):
                        suggestions = []
                except:
                    suggestions = []
            history[user_id]['description_suggestion'] = suggestions
            print("suggestions", type(suggestions), suggestions) 
            messages = [
                TextMessage(text="Okay, let's continue to improve your prompt."),
                TextMessage(text="Inorder to generate a more specific image, please provide more detailed description."),
                TextMessage(text='Also according to your query, I have some suggestions for you, you can click the following tags to add them to your prompt.' if suggestions != [] else "Feel free to type more details about the image you want to generate."),
                TemplateSendMessage(
                    alt_text='Suggestions',
                    template=CarouselTemplate(columns=[
                                CarouselColumn(
                                    text=suggestion[:117] + "..." if len(suggestion) > 120 else suggestion,
                                    actions=[
                                        MessageAction(label='I want this!', text=f"<DESCRIPTIONSAMPLE {i}>")
                                    ]
                                ) for i, suggestion in enumerate(suggestions)
                            ])
                ),
            ]
            history[user_id]["state"] = 3
            line_bot_api.reply_message(event.reply_token, messages)
    elif user_state == 2:
        if user_message == 'Continue':
            try:
                suggestions = description_advisor.invoke(history[user_id]['base_prompt'])['result']
                if not isinstance(suggestions[0], str):
                    suggestions = []
            except:
                try:
                    suggestions = description_advisor.invoke(history[user_id]['base_prompt'])['result']
                    if not isinstance(suggestions[0], str):
                        suggestions = []
                except:
                    suggestions = []
            history[user_id]['description_suggestion'] = suggestions
            print("suggestions", type(suggestions), suggestions) 
            messages = [
                TextMessage(text="Okay, let's continue to improve your prompt."),
                TextMessage(text="Inorder to generate a more specific image, please provide more detailed description."),
                TextMessage(text='Also according to your query, I have some suggestions for you, you can click the following tags to add them to your prompt.' if suggestions != [] else "Feel free to type more details about the image you want to generate."),
                TemplateSendMessage(
                    alt_text='Suggestions',
                    template=CarouselTemplate(columns=[
                                CarouselColumn(
                                    text=suggestion[:117] + "..." if len(suggestion) > 120 else suggestion,
                                    actions=[
                                        MessageAction(label='I want this!', text=f"<DESCRIPTIONSAMPLE {i}>")
                                    ]
                                ) for i, suggestion in enumerate(suggestions)
                            ])
                )
            ]
            history[user_id]["state"] = 3
            line_bot_api.reply_message(event.reply_token, messages)
        elif user_message[:len("<RAGPROMPT")] == "<RAGPROMPT":
            selected_id = int(user_message.split(" ")[1][:-1])
            messages = [
                TextMessage(text=f"Great! You have selected the prompt: \n\"{history[user_id]['rag'][selected_id]}\""),
                TextSendMessage(
                    text='If you want to add more details to the prompt, please type them below. \nOtherwise, you can click the button below to go to the next step.',
                    quick_reply=QuickReply(
                        items=[
                            QuickReplyButton(
                                action=MessageAction(label="Let's goooo!", text="start generating")
                            )]
                    ))
            ]
            history[user_id]["state"] = 5
            line_bot_api.reply_message(event.reply_token, messages)
            
    elif user_state == 3:
        if user_message[:len("<DESCRIPTIONSAMPLE")] == "<DESCRIPTIONSAMPLE":
            history[user_id]['description'] = history[user_id]['description_suggestion'][int(user_message.split(" ")[1][:-1])]
        else:
            res = finisher.invoke(user_message)['result']
            if res == 'yes':
                messages = [
                    TextMessage(text="Great! Let's generate the image!"),
                    TemplateSendMessage(
                        alt_text='Get started',
                        template=ConfirmTemplate(
                                text="Are you ready to generate the image? (It may take a while)",
                                actions=[
                                    MessageAction(
                                        label='Yes!',
                                        text='Yes!'
                                    ),
                                    MessageAction(
                                        label="I'm ready!",
                                        text="I'm ready!"
                                    )
                                ]
                            )
                    )
                ]
                history[user_id]["state"] = 6
                line_bot_api.reply_message(event.reply_token, messages)
            else:
                history[user_id]['description'] = user_message
        if res != 'yes':
            styles = style_advisor.invoke(history[user_id]['description'])['result']
            history[user_id]['suggested_styles'] = styles
            messages = [
                TextMessage(text="Great! Now, let's select some styles for your image"),
                TextMessage(text="You can select the styles below or type the style you want."),
                TextSendMessage(
                text='Here are some styles that might be helpful for you: ',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=MessageAction(label=style, text=style)
                        ) for style in history[user_id]['suggested_styles']]
                ))
            ]
            history[user_id]["state"] = 4
            line_bot_api.reply_message(event.reply_token, messages)
        
    elif user_state == 4:
        if user_message not in history[user_id]["suggested_styles"]:
            res = style_distinguisher.invoke(user_message)
            req_type = res['mode']
            req_style = res['style']
            print("req_type, req_style:", req_type, req_style)
            if req_type == "add" and req_style != "":
                history[user_id]["styles"].append(req_style)
                messages = [
                    TextMessage(text="Great! You have added the style: " + req_style),
                    TextMessage(text="Now you can select more styles."),
                    TextSendMessage(
                    text='If you think you have selected enough styles, you can ask me to generate the image.',
                    quick_reply=QuickReply(
                        items=[
                            QuickReplyButton(
                                action=MessageAction(label=style, text=style)
                            ) for style in history[user_id]['suggested_styles']]
                    ))
                ]
            elif req_type == "preview" and req_style != "":
                history[user_id]["preview_style"] = req_style
                messages = [
                    TextMessage(text=f"I can see you want to preview the image of the style: {req_style}.\n Let's generate the image!"),
                    TemplateSendMessage(
                        alt_text='Preview it?',
                        template=ConfirmTemplate(
                                text="Preview it? (It may take a while)",
                                actions=[
                                    MessageAction(
                                        label='Yes',
                                        text='Yes'
                                    ),
                                    MessageAction(
                                        label="No",
                                        text="No"
                                    )
                                ]
                            )
                    )
                ]
                history[user_id]["state"] = 8
            elif req_type == "final" or req_style == "":
                messages = [
                    TextMessage(text="Great! Let's generate the image!"),
                    TemplateSendMessage(
                        alt_text='Get started',
                        template=ConfirmTemplate(
                                text="Are you ready to generate the image? (It may take a while)",
                                actions=[
                                    MessageAction(
                                        label='Yes!',
                                        text='Yes!'
                                    ),
                                    MessageAction(
                                        label="I'm ready!",
                                        text="I'm ready!"
                                    )
                                ]
                            )
                    )
                ]
                history[user_id]["state"] = 6
            line_bot_api.reply_message(event.reply_token, messages)
        
        else:
            history[user_id]["suggested_styles"].remove(user_message)
            if history[user_id]['suggested_styles'] != []:
                messages = [
                    TextMessage(text="Great! You can select more styles or type."),
                    TextSendMessage(
                    text='If you think you have selected enough styles, you can ask me to generate the image.',
                    quick_reply=QuickReply(
                        items=[
                            QuickReplyButton(
                                action=MessageAction(label=style, text=style)
                            ) for style in history[user_id]['suggested_styles']]
                    ))
                ]
            else:
                messages = [
                    TextMessage(text="Great! I think your prompt is good enough, let's generate the image!"),
                    TemplateSendMessage(
                        alt_text='Get started',
                        template=ConfirmTemplate(
                                text="Are you ready to generate the image? (It may take a while)",
                                actions=[
                                    MessageAction(
                                        label='Yes!',
                                        text='Yes!'
                                    ),
                                    MessageAction(
                                        label="I'm ready!",
                                        text="I'm ready!"
                                    )
                                ]
                            )
                    )
                ]
                history[user_id]["state"] = 6
            line_bot_api.reply_message(event.reply_token, messages)
    
    elif user_state == 5:
        messages = [
                TextMessage(text="Great! I think your prompt is good enough, let's generate the image!"),
                TemplateSendMessage(
                    alt_text='Get started',
                    template=ConfirmTemplate(
                            text="Are you ready to generate the image? (It may take a while)",
                            actions=[
                                MessageAction(
                                    label='Yes!',
                                    text='Yes!'
                                ),
                                MessageAction(
                                    label="I'm ready!",
                                    text="I'm ready!"
                                )
                            ]
                        )
                )
                ]
        history[user_id]["state"] = 6
        line_bot_api.reply_message(event.reply_token, messages)
    
    elif user_state == 6:
        img, _ = txt2img("A beautiful sunset over the ocean.")
        url = img_post(img)
        messages = [
            TextMessage(text="Here is the image you requested:"),
            ImageSendMessage(original_content_url=url, preview_image_url=url),
            TemplateSendMessage(
                alt_text='Contributing to the community',
                template=ConfirmTemplate(
                        text="Hey, do you want to share your prompt with the community?",
                        actions=[
                            MessageAction(
                                label='Yes!',
                                text='Yes!'
                            ),
                            MessageAction(
                                label="Next time",
                                text="Next time"
                            )
                        ]
                    )
                )
            ]
        history[user_id]["state"] = 7
        line_bot_api.reply_message(event.reply_token, messages)
    
    elif user_state == 7:
        if user_message == "Yes!":
            messages = [
                TextMessage(text="Great! Your prompt has been shared with the community!"),
                TextMessage(text="Thank you for contributing!"),
                TextSendMessage(
                    text="If you have any other requests, feel free to start a new round!",
                    quick_reply=QuickReply(
                        items=[
                            QuickReplyButton(
                                action=MessageAction(label="Restart", text="Get started")
                            )]
                    ))
            ]
            
        elif user_message == "Next time":
            messages = [
                TextMessage(text="Okay!"),
                TextSendMessage(
                    text="If you have any other requests, feel free to start a new round!",
                    quick_reply=QuickReply(
                        items=[
                            QuickReplyButton(
                                action=MessageAction(label="Restart", text="Get started")
                            )]
                    ))
            ]
        history[user_id] = init_info()
        line_bot_api.reply_message(event.reply_token, messages)
    
    elif user_state == 8:
        if user_message == "Yes":
            img = cv2.imread("./Start.png")
            url = img_post(img)
            messages = [
                TextMessage(text=f"Here is the image that you can preview the style: {history[user_id]['preview_style']}"),
                ImageSendMessage(original_content_url=url, preview_image_url=url),
                TextMessage(text="Now you can continue to select more styles or generate the image."),
                TextSendMessage(
                    text='If you think you have selected enough styles, you can ask me to generate the image.',
                    quick_reply=QuickReply(
                        items=[
                            QuickReplyButton(
                                action=MessageAction(label=style, text=style)
                            ) for style in history[user_id]['suggested_styles']]
                    ))
                ]
        elif user_message == "No":
            messages = [
                TextMessage(text="Okay! Now you can continue to select more styles or generate the image."),
                TextSendMessage(
                    text='If you think you have selected enough styles, you can ask me to generate the image.',
                    quick_reply=QuickReply(
                        items=[
                            QuickReplyButton(
                                action=MessageAction(label=style, text=style)
                            ) for style in history[user_id]['suggested_styles']]
                    ))
            ]
        history[user_id]["state"] = 4
        line_bot_api.reply_message(event.reply_token, messages)

    else:
        print("Invalid user state")
        
        
    if user_message == "hi":
        messages = [
            TextMessage(text="Hii, nice to meet you!"),
        ]
        line_bot_api.reply_message(event.reply_token, messages)
        

if __name__ == "__main__":
    app.run()
    # app.run(port=5000, debug=True)