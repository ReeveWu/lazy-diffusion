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

from linebot.models import MessageAction, TemplateSendMessage, ConfirmTemplate

from collections import defaultdict

from utils.LLM import GenerateDescriptions, GenerateStyle, RetrievalWithPrompt

retriever = RetrievalWithPrompt(mode=1)
description_advisor = GenerateDescriptions()
style_advisor = GenerateStyle()


load_dotenv()

CHANNEL_ACCESS_TOKEN = os.getenv('CHANNEL_ACCESS_TOKEN')
CHANNEL_SECRET = os.getenv('CHANNEL_SECRET')

def init_info():
    return {
        "state": None,
        "see_suggestions": False,
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
# state 5: user has finished selecting styles and is ready to generate the image
# state None: user has not started a round

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
    
    if user_state is None:
        if user_message == "Let's go!":
            messages = [
                TextMessage(text="Great! Let's start with a base prompt. "), 
                TextMessage(text="Please type a sentence that describes the image you want to generate."),
                TextMessage(text="p.s. If you have not decided yet, you can click 'Not now!' to exit.")
            ]
            history[user_id]["state"] = 0
            line_bot_api.reply_message(event.reply_token, messages)
            
        elif user_message == "Not now!":
            messages = [
                TextMessage(text="Okay, let me know when you're ready!"),
            ]
            line_bot_api.reply_message(event.reply_token, messages)
            
        elif user_message == "Get start":
            messages = [
                TextMessage(text="Hi, I'm your image generation assistant. "),
                TextMessage(text="I will guide you step by step to improve your prompt for generating high-quality images."),
                TextMessage(text="Click the button below to embark on this magical journey!"),
                TemplateSendMessage(
                    alt_text='Get start',
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
            print("docs", docs)
            messages = [
                TextMessage(text="Here are some prompts that might be helpful for you: "),
                TemplateSendMessage(
                    alt_text='Suggestions',
                    template=CarouselTemplate(columns=[
                                CarouselColumn(
                                    text=doc.page_content,
                                    actions=[
                                        URIAction(label='I want this!', uri=doc.metadata['url'])
                                    ]
                                ) for doc in docs
                            ])
                )
            ]
            history[user_id]["state"] = 2
            line_bot_api.reply_message(event.reply_token, messages)
            
        elif user_message == "No":
            suggestions = description_advisor.invoke(history[user_id]['base_prompt'])['result']
            print("suggestions", type(suggestions), suggestions)
            messages = [
                TextMessage(text="Okay, let's continue to improve your prompt."),
                TextMessage(text="Inorder to generate a more specific image, please provide more detailed description."),
                TextMessage(text='Also according to your query, I have some suggestions for you, you can click the following tags to add them to your prompt.'),
                TemplateSendMessage(
                    alt_text='Suggestions',
                    template=CarouselTemplate(columns=[
                                CarouselColumn(
                                    text=suggestion,
                                    actions=[
                                        MessageAction(label='I want this!', text=suggestion)
                                    ]
                                ) for suggestion in suggestions
                            ])
                )
            ]
            history[user_id]["state"] = 3
            line_bot_api.reply_message(event.reply_token, messages)
    elif user_state == 2:
        pass
    elif user_state == 3:
        history[user_id]['description'] = user_message
        styles = style_advisor.invoke(history[user_id]['description'])['result']
        # styles = ['Minimalist', 'Vibrant', 'Impressionist', 'Cubist', 'Cartoonish', 'Watercolor']
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
        if False:
            pass
        elif False:
            pass
        else:
            history[user_id]["styles"].append(user_message)
            if user_message in history[user_id]["suggested_styles"]:
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
                    alt_text='Get start',
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
                history[user_id]["state"] = 5
            line_bot_api.reply_message(event.reply_token, messages)
    else:
        print("Invalid user state")
        
        
    if user_message == "hi":
        messages = [
            TextMessage(text="Hii, nice to meet you!"),
        ]
        line_bot_api.reply_message(event.reply_token, messages)
        

if __name__ == "__main__":
    # app.run()
    app.run(port=5000, debug=True)