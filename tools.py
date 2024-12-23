import os
import google.generativeai as genai
import json
import math
import requests
import sys
import time
import threading
import wave
import pyaudio

import numpy as np
import soundfile as sf

from typing_extensions import TypedDict
from typing import Any, Dict, List, Optional, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import yaml
import copy

from utils import save_audio, play_audio_with_pyaudio, invoke_chatbot, unscripted_pronunciation_assessment_continuous_from_file
from utils import AgentState

import azure.cognitiveservices.speech as speechsdk

## Load keys and config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load the API key
os.environ["GOOGLE_API_KEY"] = config["google_cloud_info"]["google_cloud_api_key"]
speech_key = config["azure_info"]["azure_speech_key"]
service_region = config["azure_info"]["azure_service_region"]

# Define tools
def stt(state: AgentState) -> AgentState:
    """
    Convert speech to text in real-time from microphone input.
    """
    result_text = []
    updated_state = copy.deepcopy(state)
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

    updated_state["action_num"] += 1

    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    done = False
    audio_args = state["audio_params"]

    def stop_cb(evt: speechsdk.SessionEventArgs):
        """Callback that signals to stop continuous recognition upon receiving an event `evt`."""
        nonlocal done
        done = True

    speech_recognizer.recognized.connect(lambda evt: result_text.append(evt.result.text))
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    speech_recognizer.start_continuous_recognition()
    print("Please speak into the microphone...")

    global recording_event
    filename = f"input_audio_{state["action_num"]}.wav"
    recording_event = threading.Event()
    recording_event.set()
    audio_thread = threading.Thread(target=save_audio, args=(filename, audio_args["format_"], audio_args["channels"], audio_args["sample_rate"], audio_args["chunk"], recording_event))
    audio_thread.start()

    start_time = time.time()

    while not done:
        time.sleep(0.5)
        if time.time() - start_time > state["max_speaking_time"] + 1:
            print("Thank you for speaking.\n Processing info.....")
            speech_recognizer.stop_continuous_recognition()
            done = True

    recording_event.clear()
    audio_thread.join()

    speech_recognizer.stop_continuous_recognition()

    text = " ".join(result_text)
    messages = state["messages"]
    messages.append(HumanMessage(text))
    updated_state["messages"] = messages
    
    updated_state["actions"].append("Speech-to-Text")
    time.sleep(1)
    return updated_state

def chat(state: AgentState) -> AgentState:
    """
    Simulate Chat.
    Takes text input and generates a response.
    """
    messages = state["messages"]
    messages, input_tokens, output_tokens = invoke_chatbot(messages, temperature=0)
    updated_state = copy.deepcopy(state)

    updated_state["messages"] = messages
    updated_state["actions"].append("Chat")
    updated_state["total_input_tokens"] += input_tokens
    updated_state["total_output_tokens"] += output_tokens
    updated_state["token_usage_history"].append({
        "action": f"chat_{updated_state["action_num"]}",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    })

    return updated_state

def tts(state: AgentState) -> AgentState:
    output_path = f"output_audio_{state['action_num']}.wav"
    last_ai_message = state["messages"][-1].content
    print(last_ai_message)
    time.sleep(1)
    ssml_text = (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        f'<voice name="en-US-JennyNeural">{last_ai_message}</voice>'
        '</speak>'
    )
    sample_rate = state["audio_params"]["sample_rate"]

    file_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=file_config)
    result = speech_synthesizer.speak_ssml_async(ssml_text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        # Optional: strip_end_silence(output_path, sample_rate=sample_rate)
        play_audio_with_pyaudio(output_path)
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))

    updated_state = copy.deepcopy(state)

    updated_state["actions"].append("Text-to-Speech")

    return updated_state
    
def check_conversation_end(state: AgentState): 
    """
    Pass the message history to the model and check if the conversation should end
    """
    print("Conditional Node: Checking if conversation should end...")
    stop_cond = input("Enter 'yes' to end the conversation: ")
    time.sleep(1)
    if "yes" in stop_cond.lower(): 
        return "__end__"
    else:
        return "stt"
    
def pronunciation_assessment(state: AgentState) -> AgentState:
    """
    Pronunciation assessment from a file.
    """
    # Load the audio file
    updated_state = copy.deepcopy(state)
    filename = f"input_audio_{state["action_num"]}.wav"

    time.sleep(5)
    scores, scores_by_part = unscripted_pronunciation_assessment_continuous_from_file(filename)

    ## store scores for analysis in a temporary file in database
    pron_score = {
        'action': state["action_num"],
        'scores': scores
    }

    if os.path.exists("pronunciation_scores.json"):
        with open("pronunciation_scores.json", "a") as file:
            json.dump(pron_score, file)
    else:
        with open("pronunciation_scores.json", "w") as file:
            json.dump(pron_score, file)

    updated_state["actions"].append("Pronunciation Assessment")

    return updated_state

def grammar_assessment(state: AgentState) -> AgentState:
    """
    Grammar assessment from a file.
    """
    # Load the audio file
    updated_state = copy.deepcopy(state)
    filename = f"input_audio_{state["action_num"]}.wav"

    ## perform grammar assessment

    updated_state["actions"].append("Grammar Assessment")

    return updated_state

def fluency_aasssessment(state: AgentState) -> AgentState:
    """
    Fluency assessment from a file.
    """
    # Load the audio file
    updated_state = copy.deepcopy(state)
    filename = f"input_audio_{state["action_num"]}.wav"

    ## perform fluency assessment

    updated_state["actions"].append("Fluency Assessment")

    return updated_state