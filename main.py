############################ STREAMLIT APP ########################################
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from io import BytesIO
import pyttsx3
from pydub import AudioSegment
import base64, os
import numpy as np
from rag.create_embedding import embed_texts
from rag.create_embedding import ExtractCreateEmbeddings
from rag.similarity_search import SimilaritySearch
import warnings
from llm.chat import llm_call
import json 
from google.cloud import texttospeech
from typing import List

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech


#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "tect-to-speech-sa-key.json"
PROJECT_ID = "walmart-retail-media"

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'embed_vectors' not in st.session_state:
    st.session_state['embed_vectors'] = [] 
    
if 'msg' not in st.session_state:
    st.session_state['msg'] = [] 


# Suppress all warnings
warnings.filterwarnings("ignore")

# Create a folder to save the uploaded files
SAVE_FOLDER = 'data'
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Save the uploaded file
if uploaded_file is not None:
    # Define the file path
    file_path = os.path.join(SAVE_FOLDER, uploaded_file.name)

    # Save the file to the folder
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File has been successfully uploaded")
    
    if 'extracted_text' not in st.session_state:
        embed = ExtractCreateEmbeddings(file_path) 
        st.session_state['extracted_text'] = embed.extract_text_from_pdf()  
    
        embedded_vectors = []
        for i in range(len(st.session_state['extracted_text'])):
            embedded_vectors.append(st.session_state['extracted_text'][i][1])

        st.session_state['embed_vectors'] = np.array(embedded_vectors)
################################################################################################################################
##############################################   RAG IMPLEMENTATION  ###########################################################
ss = SimilaritySearch()




################################################################################################################################

# Function to transcribe audio using SpeechRecognition (Google Web Speech API or offline engine)
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            # Using Google Web Speech API (free version)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            return "Sorry, there was an error with the Speech API."
        
# Function for Google Cloud Text-to-Speech
def text_to_speech_google_cloud(text, language_code="en-IN", voice_name="en-IN-Wavenet-C"):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name,  # Use WaveNet voices for natural, human-like output
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    audio_bytes = BytesIO(response.audio_content)
    return audio_bytes




def text_to_speech_pyttsx3(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 120)  # Adjust speed (default ~200)
    
    audio_bytes = BytesIO()

    # Save output as a file and read it into memory
    engine.save_to_file(text, "output.mp3")
    engine.runAndWait()  # Ensure processing is complete
    
    with open("output.mp3", "rb") as f:
        audio_bytes.write(f.read())

    audio_bytes.seek(0)
    return audio_bytes

def autoplay_audio(audio_bytes):
    print("INSIDE AUTOPLAY")
    # Encode the audio bytes to base64
    audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
    
    audio_html = f'<audio controls autoplay="true" src="data:audio/wav;base64,{audio_base64}">'
    st.markdown(audio_html, unsafe_allow_html=True)

def transcribe_multiple_languages_v2(
    audio_file: str,
    language_codes: List[str],
) -> cloud_speech.RecognizeResponse:
    """Transcribe an audio file using Google Cloud Speech-to-Text API with support for multiple languages.
    Args:
        audio_file (str): Path to the local audio file to be transcribed.
            Example: "resources/audio.wav"
        language_codes (List[str]): A list of BCP-47 language codes to be used for transcription.
            Example: ["en-US", "fr-FR"]
    Returns:
        cloud_speech.RecognizeResponse: The response from the Speech-to-Text API containing the
            transcription results.
    """
    client = SpeechClient()

    # Reads a file as bytes
    with open(audio_file, "rb") as f:
        audio_content = f.read()

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=language_codes,
        model="latest_long",
    )

    request = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/global/recognizers/_",
        config=config,
        content=audio_content,
    )

    # Transcribes the audio into text
    response = client.recognize(request=request)
    # Prints the transcription results
    for result in response.results:
        print(f"Transcript: {result.alternatives[0].transcript}")

    return response

# Main function


def main():
    st.sidebar.title("Voice Chatbot")

    st.title("Voice Chatbot")
    if st.sidebar.button("Terminate Session"):
        st.write("Session terminated!")
        st.stop()
    if uploaded_file is None:
        st.write("Please upload a file to enable voice input.")
    else:
        st.write("Hi there! Click on the voice recorder to interact with me. How can I assist you today?")

        tab1, tab2 = st.tabs(["Voice Interaction", "Conversation History"])
        

        # Tab 1: Voice Interaction
        with tab1:
            recorded_audio = audio_recorder(key=f'audio_recorder') 
                
            if recorded_audio:
                audio_file_path = f"audio.wav"  # Use a unique file name for each iteration
                with open(audio_file_path, "wb") as f:
                    f.write(recorded_audio)

                # Convert the audio to .wav format if it's not already
                audio_data = AudioSegment.from_file(audio_file_path)
                audio_data.export(audio_file_path, format="wav")
                # Transcribe the audio to text using SpeechRecognition
                print("Before Speech to text")
                transcribed_text = transcribe_audio(audio_file_path)
                print("After Speech to text")
                print(transcribe_audio)
                print("Before Question Embedding")
                target_vector = np.array(embed_texts(transcribed_text)[0])
                print("After Question Embedding")


                context_pages = ss.search(target_vector=target_vector, vectors=st.session_state['embed_vectors'])

                context_content = {}
                for i in context_pages:
                    context_content[f"page_{i+1}"] = st.session_state['extracted_text'][i][0]
                rag_search = f"CONTEXT: {context_content}"
                #print(rag_search)
                transribed_rag_search = transcribed_text + "\n" + rag_search 
                input_msg={'author':'user','message':transcribed_text}
                st.session_state['msg'].append(input_msg)
                #This will contain the question to be asked
                rag_question = {"role": "user", "parts": [{"text" : transribed_rag_search}]}
                question = st.session_state['history'] + [rag_question] 

                st.session_state['history'].append({"role": "user", "parts": [{"text" : transcribed_text}]})
                # Get AI response (chatbot logic)
                chat_response = llm_call(message=question)
                json_res = json.loads(chat_response)
                ai_response = json_res[0]["agent_response"]
                print("ai response", ai_response)
                st.session_state['history'].append({"role": "model", "parts": [{"text": chat_response}]})
                output_msg={'author':'assistant','message':ai_response}
                st.session_state['msg'].append(output_msg)
                #st.session_state['output'].append(ai_response)
                st.html(
                        """
                    <style>
                        .stChatMessage:has(.chat-user) {
                            flex-direction: row-reverse;
                            text-align: right;
                        }
                    </style>
                    """
                    )  
                # Convert AI response to speech using gTTS
                print("Before text to speech")
                # tts_audio = text_to_speech_pyttsx3(ai_response)
                tts_audio = text_to_speech_google_cloud(ai_response) 
                print("After text to speech")
                 
                autoplay_audio(tts_audio)
                # Play the generated audio response with autoplay
                

        with tab2:
            for message in st.session_state['msg']:
                    with st.chat_message(message["author"]):                              
                
                        st.html(f"<span class='chat-{message['author']}'></span>")
                        st.write(message["message"])
            
            
            
            # Print conversation history in the console
            print("Conversation History  :::::  ", st.session_state['history'])
                
                
if __name__ == "__main__":
    main()