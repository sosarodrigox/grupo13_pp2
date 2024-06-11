import streamlit as st
from st_audiorec import st_audiorec
import speech_recognition as sr

st.title("Grabador y Transcriptor de Audio")

# Iniciar el grabador de audio
audio_data = st_audiorec()

# Verificar si hay datos de audio
if audio_data is not None:
    # Guardar el audio en un archivo
    audio_file_path = "audio.wav"
    with open(audio_file_path, "wb") as f:
        f.write(audio_data)

    # Transcribir el audio
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)
            texto_transcrito = recognizer.recognize_google(audio, language="es-ES")
            st.write(f"El libro detectado es: {texto_transcrito}")
    except sr.UnknownValueError:
        st.write("No se pudo entender el audio.")
    except sr.RequestError as e:
        st.write(f"Error en la solicitud; {e}")
