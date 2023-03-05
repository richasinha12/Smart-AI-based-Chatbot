#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# for speech to text
get_ipython().system('pip install SpeechRecognition  ')
# for text to speech
get_ipython().system('pip install gTTS  ')
# for language modelimport numpy as np

get_ipython().system('pip install transformers  ')
get_ipython().system('pip install tensorflow ')


# In[ ]:


import numpy as np


# In[ ]:


get_ipython().system('pip install PyAudio')


# In[ ]:


## for language model
import transformers

## for data
#import os
import datetime
import numpy as np


# In[ ]:


from gtts import gTTS
import os


# In[ ]:


import speech_recognition as sr


# In[ ]:


pip install pyttsx3


# In[ ]:


import pyttsx3


# In[ ]:


import datetime
import numpy as np
import os

from gtts import gTTS

import speech_recognition as sr

import pyttsx3

import transformers

# Import oneDNN and Intel SEAPI libraries
import onednn as dnnl
import intel_seapi as iseapi

class ChatBot():
    def __init__(self, name):
        print("--- starting up", name, "---")
        self.name = name
        
    def speech_to_text(self):
        # Initialize Intel SEAPI speech recognizer
        recognizer = iseapi.SpeechRecognizer()

        # Set up microphone input stream
        mic = sr.Microphone()

        # Start recording and processing audio
        with mic as source:
            recognizer.start()
            recognizer.process_audio_stream(source)

        # Retrieve recognized text
        self.text = recognizer.get_recognition()

        print("me --> ", self.text)

    @staticmethod
    def text_to_speech(text):
        print("ai --> ", text)

        # Generate speech using gTTS and save to temporary file
        tts = gTTS(text=text, lang='en')
        tts.save("temp.mp3")

        # Initialize Intel SEAPI text-to-speech converter
        engine = iseapi.TextToSpeechEngine()

        # Load speech from temporary file and play through system default audio output
        with open("temp.mp3", "rb") as f:
            engine.play(f.read(), blocking=True)

        # Delete temporary file
        os.remove("temp.mp3")

    def wake_up(self, text):
        return True if self.name in text.lower() else False
    
    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')

# Main function
if __name__ == "__main__":
    ai = ChatBot(name="maya")

    # Load DialoGPT model using transformers
    nlp = transformers.pipeline("conversational", model="microsoft/DialoGPT-medium")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    while True:
        ai.speech_to_text()

        ## wake up
        if ai.wake_up(ai.text) is True:
            res = "Hello I am Maya the AI, what can I do for you?"

        ## action time
        elif "time" in ai.text:
            res = ai.action_time()

        ## respond politely
        elif any(i in ai.text for i in ["thank","thanks"]):
            res = np.random.choice(["you're welcome!","anytime!","no problem!","cool!","I'm here if you need me!","peace out!"])

        ## conversation
        else:   
            chat = nlp(transformers.Conversation(ai.text), pad_token_id=50256)
            res = str(chat)
            res = res[res.find("bot >> ")+6:].strip()

        ai.text_to_speech(res)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




