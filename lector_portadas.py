import streamlit as st
from st_audiorec import st_audiorec
import speech_recognition as sr
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pytesseract
from PIL import Image
import re


# Función para limpiar texto
def limpiar_texto(texto):
    return re.sub(r"[^a-zA-Z\s]", "", texto).strip()


# Función para encontrar coincidencias cercanas entre dos listas
def encontrar_coincidencias(lista1, lista2, umbral=0.25):
    coincidencias = []
    for item in lista1:
        coincidencias_cercanas = difflib.get_close_matches(
            item, lista2, n=1, cutoff=umbral
        )
        if coincidencias_cercanas:
            coincidencias.append((item, coincidencias_cercanas[0]))
    return coincidencias


# Carga de datos
url = "https://raw.githubusercontent.com/sosarodrigox/grupo13_pp2/main/data/data.csv"
data = pd.read_csv(url)

# Preprocesamiento
data[data.select_dtypes(include=["object"]).columns] = data.select_dtypes(
    include=["object"]
).fillna("")

libros = (
    data["title"]
    + " "
    + data["subtitle"]
    + " "
    + data["categories"]
    + " "
    + data["authors"]
    + " "
    + data["published_year"].astype(str)
)

# Vectorización TF-IDF
vector_tfidf = TfidfVectorizer()
vector_caracteristicas = vector_tfidf.fit_transform(libros)

# Cálculo similitud de coseno
similitud = cosine_similarity(vector_caracteristicas, vector_caracteristicas)


# Función para recomendar libros
def recomendar_libros(nombre_libro):
    st.write(f"Libro detectado: {nombre_libro}")
    lista_titulos_completa = data["title"].tolist()
    encontrar_cercanos = difflib.get_close_matches(nombre_libro, lista_titulos_completa)

    if encontrar_cercanos:
        cercanos = encontrar_cercanos[0]
        indice_de_libro = data[data.title == cercanos].index[0]
        puntaje_similitud = list(enumerate(similitud[indice_de_libro]))
        libros_similares_ordenados = sorted(
            puntaje_similitud, key=lambda x: x[1], reverse=True
        )

        recomendaciones = []
        for i, (index, score) in enumerate(libros_similares_ordenados[1:11], 1):
            libro = {
                "Indice": i,
                "Titulo": data.iloc[index]["title"],
                "Autor": data.iloc[index]["authors"],
                "Año": data.iloc[index]["published_year"],
                "Categoria": data.iloc[index]["categories"],
                "Similitud": f"{score * 100:.2f}%",
            }
            recomendaciones.append(libro)

        df_recomendaciones = pd.DataFrame(recomendaciones)
        return df_recomendaciones

    else:
        st.write("No se encontró ninguna coincidencia para el libro ingresado.")
        return pd.DataFrame()


# Función de OCR usando Tesseract
def ocr_tesseract(image_path):
    image = Image.open(image_path)
    texto = pytesseract.image_to_string(image, lang="spa")  # 'spa' para español
    return texto


# Interfaz de usuario
st.title("Grabador y Transcriptor de Audio")

# Sección de grabación de audio
audio_data = st_audiorec()

# Verificar si hay datos de audio
if audio_data is not None:
    audio_file_path = "audio.wav"
    with open(audio_file_path, "wb") as f:
        f.write(audio_data)

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)
            texto_transcrito = recognizer.recognize_google(audio, language="es-ES")
            st.write(f"El libro detectado es: {texto_transcrito}")

            df_recomendaciones = recomendar_libros(texto_transcrito)
            if not df_recomendaciones.empty:
                st.write("Libros similares encontrados:")
                st.dataframe(df_recomendaciones)
            else:
                st.write("Lo sentimos, no se encontraron libros similares.")

    except sr.UnknownValueError:
        st.write("No se pudo entender el audio.")
    except sr.RequestError as e:
        st.write(f"Error en la solicitud; {e}")

# Sección de subida y procesamiento de imágenes
st.header("Sube una imagen de la portada de un libro")
imagen = st.file_uploader("Sube tu foto aquí:")

if imagen:
    with open("uploaded_image.jpg", "wb") as f:
        f.write(imagen.getbuffer())
    st.success("Imagen subida correctamente!")

    portada = Image.open("uploaded_image.jpg")
    st.image(portada, width=200)

    lectura_texto = ocr_tesseract("uploaded_image.jpg").upper()
    libro = [lectura_texto]

    libro_str = libro[0][1:-1]
    pattern = re.compile(r"\[[^\]]*\]|\d+\.\d+|\d+|[A-Z][A-Z ]*[A-Z]")
    matches = pattern.findall(libro_str)
    result = []
    for match in matches:
        if re.match(r"\d+\.\d+", match):
            result.append(float(match))
        elif re.match(r"\d+", match):
            result.append(int(match))
        else:
            result.append(match.strip())

    libro_limpio = [
        limpiar_texto(str(item)) for item in result if limpiar_texto(str(item))
    ]
    libro_usuario = [item.upper() for item in libro_limpio]

    df = pd.read_csv(url, header=0, encoding="latin-1")
    titles = [title.upper().strip() for title in df["title"].tolist()]

    coincidencias = encontrar_coincidencias(libro_usuario, titles, umbral=0.75)

    if coincidencias:
        coincidencia_index = max(
            coincidencias, key=lambda tupla: len(max(tupla, key=len))
        )
        coincidencia_mas_relevante = coincidencia_index[1]

        st.write("Se encontraron las siguientes coincidencias:")
        df_resultado = df[
            df["title"].str.upper().str.strip() == coincidencia_mas_relevante
        ]
        df_resultado = df_resultado.drop(columns=["portada"])
        df_resultado = df_resultado.reset_index(drop=True)
        df_resultado.columns = [col.upper() for col in df_resultado.columns]

        st.dataframe(df_resultado, height=600)
    else:
        st.warning("No se encontraron coincidencias.")
else:
    st.warning("Por favor, carga una imagen.")
