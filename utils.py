import os
import time
import wave
import pyaudio

import numpy as np
import soundfile as sf

from typing_extensions import TypedDict
from typing import Any, Dict, List, Optional, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import azure.cognitiveservices.speech as speechsdk
import json
import yaml

## Define environment variables
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load the API key
os.environ["GOOGLE_API_KEY"] = config["google_cloud_info"]["google_cloud_api_key"]
speech_key = config["azure_info"]["azure_speech_key"]
service_region = config["azure_info"]["azure_service_region"]

## Define the types
class TokenCount(TypedDict):
    action: str
    input_tokens: int
    output_tokens: int

class AudioParams(TypedDict):
    sample_rate: int
    channels: int
    sample_width: int
    format_: int   
    chunk: int

class AgentState(TypedDict):
    messages: List[str]
    actions: List[str]
    action_num: int
    max_speaking_time: int ## contains the maximum speaking time in seconds
    audio_params: AudioParams
    token_usage_history: List[TokenCount]
    total_input_tokens: int
    total_output_tokens: int


## util functions
def read_wave_header(file_path):
    with wave.open(file_path, 'rb') as audio_file:
        framerate = audio_file.getframerate()
        bits_per_sample = audio_file.getsampwidth() * 8
        num_channels = audio_file.getnchannels()
        return framerate, bits_per_sample, num_channels
    
def push_stream_writer(stream, filenames, merged_audio_path, sample_rate, channels, sample_width): 
    byte_data = b""
    # The number of bytes to push per buffer
    n_bytes = 3200
    try:
        for filename in filenames:
            wav_fh = wave.open(filename)
            # Start pushing data until all data has been read from the file
            try:
                while True:
                    frames = wav_fh.readframes(n_bytes // 2)
                    if not frames:
                        break
                    stream.write(frames)
                    byte_data += frames
                    time.sleep(.1)
            finally:
                wav_fh.close()
        with wave.open(merged_audio_path, 'wb') as wave_file:
            wave_file.setnchannels(channels)
            wave_file.setsampwidth(sample_width)
            wave_file.setframerate(sample_rate)
            wave_file.writeframes(byte_data)
    finally:
        stream.close()

def merge_wav(audio_list, output_path, sample_rate, tag=None): 
    combined_audio = np.empty((0,))
    for audio in audio_list:
        y, _ = sf.read(audio, dtype="float32")
        combined_audio = np.concatenate((combined_audio, y))
        os.remove(audio)
    sf.write(output_path, combined_audio, sample_rate)
    if tag:
        print(f"Save {tag} to {output_path}")

def get_mispronunciation_clip(offset, duration, save_path, merged_audio_path, sample_rate, reduced_unit): 
    y, _ = sf.read(
        merged_audio_path,
        start=int((offset) / reduced_unit * sample_rate),
        stop=int((offset + duration) / reduced_unit * sample_rate),
        dtype=np.float32
    )
    sf.write(save_path, y, sample_rate)

def strip_end_silence(file_path, sample_rate):
    y, _ = sf.read(file_path, start=0, stop=-int(sample_rate*0.8), dtype=np.float32)
    sf.write(file_path, y, sample_rate)

def save_audio(filename, format_, channels, rate, chunk, recording_event):
    """
    Save audio from the microphone in parallel using PyAudio.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=format_, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    frames = []

    print("Recording audio...")
    start_time = time.time()    

    while recording_event.is_set():
        data = stream.read(chunk)
        frames.append(data)
    print("Stopping audio recording...")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded frames as a .wav file
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format_))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    print(f"Audio saved to {filename}")

    strip_end_silence(filename, rate)

def play_audio_with_pyaudio(wav_file_path):
    # Open the WAV file
    wf = wave.open(wav_file_path, 'rb')

    # Create a PyAudio instance
    p = pyaudio.PyAudio()

    # Open a stream for playback
    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True
    )

    # Read data in chunks
    chunk = 1024
    data = wf.readframes(chunk)

    # Play the sound by writing audio data to the stream
    while data:
        stream.write(data)
        data = wf.readframes(chunk)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Close PyAudio
    p.terminate()

    # Close the WAV file
    wf.close()

def invoke_chatbot(messages, temperature):
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=temperature)
    response = llm.invoke(messages)
    content = response.content
    input_tokens = response.usage_metadata["input_tokens"]
    output_tokens = response.usage_metadata["output_tokens"]
    messages.append(AIMessage(content=content))
    
    return messages, input_tokens, output_tokens

def unscripted_pronunciation_assessment_continuous_from_file(filename):
    """Performs continuous pronunciation assessment asynchronously with input from an audio file.
        See more information at https://aka.ms/csspeech/pa"""

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    audio_config = speechsdk.audio.AudioConfig(filename=filename)

    # Create pronunciation assessment config, set grading system, granularity and if enable miscue based on your requirement.
    enable_miscue = False
    enable_prosody_assessment = True
    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=enable_miscue)
    
    ## can also include topic-based content assessment if required
    if enable_prosody_assessment:
        pronunciation_config.enable_prosody_assessment()

    # Creates a speech recognizer using a file as audio input.
    language = 'en-US'
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, language=language, audio_config=audio_config)
    # Apply pronunciation assessment config to speech recognizer
    pronunciation_config.apply_to(speech_recognizer)

    done = False
    recognized_words = []
    prosody_scores = []
    fluency_scores = []
    durations = []
    startOffset = 0
    endOffset = 0

    scores_by_part = []

    def stop_cb(evt: speechsdk.SessionEventArgs):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True

    def recognized(evt: speechsdk.SpeechRecognitionEventArgs):
        global scores_by_part
        print("pronunciation assessment for: {}".format(evt.result.text))
        pronunciation_result = speechsdk.PronunciationAssessmentResult(evt.result)

        res = {
            "accuracy_score": pronunciation_result.accuracy_score,
            "prosody_score": pronunciation_result.prosody_score,
            "pronunciation_score": pronunciation_result.pronunciation_score,
            "fluency_score": pronunciation_result.fluency_score
        }
        scores_by_part.append(res)
        nonlocal recognized_words, prosody_scores, fluency_scores, durations, startOffset, endOffset
        recognized_words += pronunciation_result.words
        fluency_scores.append(pronunciation_result.fluency_score)
        if pronunciation_result.prosody_score is not None:
            prosody_scores.append(pronunciation_result.prosody_score)
        json_result = evt.result.properties.get(speechsdk.PropertyId.SpeechServiceResponse_JsonResult)
        jo = json.loads(json_result)
        nb = jo["NBest"][0]
        durations.extend([int(w["Duration"]) + 100000 for w in nb["Words"]])
        if startOffset == 0:
            startOffset = nb["Words"][0]["Offset"]
        endOffset = nb["Words"][-1]["Offset"] + nb["Words"][-1]["Duration"] + 100000

    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognized.connect(recognized)

    # (Optional) get the session ID
    speech_recognizer.session_started.connect(lambda evt: print(f"SESSION ID: {evt.session_id}"))
    speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))

    # Stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Start continuous pronunciation assessment
    speech_recognizer.start_continuous_recognition()
    start_time = time.time()
    while not done:
        time.sleep(.5)

    speech_recognizer.stop_continuous_recognition()

    final_words = recognized_words

    durations_sum = sum([d for w, d in zip(recognized_words, durations) if w.error_type == "None"])

    # We can calculate whole accuracy by averaging
    final_accuracy_scores = []
    for word in final_words:
        if word.error_type == 'Insertion':
            continue
        else:
            final_accuracy_scores.append(word.accuracy_score)
    accuracy_score = sum(final_accuracy_scores) / (len(final_accuracy_scores)+0.001)

    # Re-calculate the prosody score by averaging
    if len(prosody_scores) == 0:
        prosody_score = float("nan")
    else:
        prosody_score = sum(prosody_scores) / len(prosody_scores)
        
    # Re-calculate fluency score
    fluency_score = 0
    if startOffset > 0:
        fluency_score = durations_sum / (endOffset - startOffset) * 100

    scores = {
        "accuracy_score": accuracy_score,
        "prosody_score": prosody_score,
        "fluency_score": fluency_score
    }

    return scores, scores_by_part