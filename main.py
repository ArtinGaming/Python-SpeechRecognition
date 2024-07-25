import openai
import pyttsx3
import speech_recognition as sr
import whisper
import pygame
import os
from dotenv import load_dotenv

load_dotenv()

audio_file = "input.wav"

if os.path.exists(audio_file):
    print(f"File '{audio_file}' found.")
    # Proceed with loading and processing the audio file
else:
    print(f"Error: File '{audio_file}' not found.")

openai.api_key = os.getenv("OPENAI_KEY")

engine = pyttsx3.init()

model = whisper.load_model("base")

turn_on = False

def play_music(mp3File):
    pygame.mixer.init()
    pygame.mixer.music.load(mp3File)
    pygame.mixer.music.play()

def transcribe_audio_to_text(filename):
    recognizer = sr.Recognizer()

    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language='fr-CA')
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

def generate_response(prompt):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=400,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response["choices"][0]["text"]

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def answer_question():
    turn_on = True
    while turn_on:
        with sr.Microphone() as source:
            play_music("StartSoundEffect.wav")
            print("Say Your Question")
            recognizer = sr.Recognizer()
            recognizer.adjust_for_ambient_noise(source)
            recognizer.dynamic_energy_threshold = 3000

            audio = recognizer.listen(source)
            filename = "input.wav"

            with open(filename, "wb") as f:
                f.write(audio.get_wav_data())

            audio = whisper.load_audio("input.wav")

            text_dict = model.transcribe(audio, fp16=False, language='English')

            text = text_dict['text'].strip()

            print(text)

            if text:
                print(f"You said: " + text)
                response = generate_response(text + " in two sentences")
                print(f"GPT-3 says: {response}")
                speak_text(response)
                turn_on = False

def main():
    try:
        with sr.Microphone() as source:
            recognizer = sr.Recognizer()
            recognizer.adjust_for_ambient_noise(source)
            recognizer.dynamic_energy_threshold = 3000

            print("Listening... ")
            speak_text("Listening")
            audio = recognizer.listen(source)
            print(audio)

            response = recognizer.recognize_google(audio)  # , language='fr-CA')

    except sr.UnknownValueError:
        print("Sorry, I couldn't Understand that.")
        speak_text("Sorry, I couldn't understand that.")

    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    

    print("Say 'Robot' to start recording your question...")
    print(response)

    # Check if the word "robot" is in the response
    if "robot" in response.lower():
        print("ROBOT RECOGNIZED!")
        answer_question()

if __name__ == "__main__":
    print("Starting")
    speak_text("Starting")
    main()