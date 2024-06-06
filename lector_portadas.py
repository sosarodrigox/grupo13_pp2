import streamlit as st
import cv2
import easyocr
import re
import pandas as pd
from difflib import get_close_matches

# Función para limpiar texto
def limpiar_texto(texto):
    return re.sub(r'[^a-zA-Z\s]', '', texto).strip()

# Función para encontrar coincidencias cercanas entre dos listas
def encontrar_coincidencias(lista1, lista2, umbral=0.25):
    coincidencias = []
    for item in lista1:
        coincidencias_cercanas = get_close_matches(item, lista2, n=1, cutoff=umbral)
        if coincidencias_cercanas:
            coincidencias.append((item, coincidencias_cercanas[0]))
    return coincidencias

# Carga la imagen con Streamlit
imagen = st.file_uploader(label='Sube tu foto aquí:')

if imagen:
    with open("uploaded_image.jpg", "wb") as f:
        f.write(imagen.getbuffer())
    st.success('Imagen subida correctamente!')

    portada = cv2.imread("uploaded_image.jpg")
    st.image(portada, width=200, channels='BGR')

    # OCR
    ocr = easyocr.Reader(['es'])
    lectura = ocr.readtext(portada)
    lectura_texto = " ".join([item[1] for item in lectura]).upper()
    libro = [lectura_texto]

    # Procesamiento del texto
    libro_str = libro[0][1:-1]
    pattern = re.compile(r'\[[^\]]*\]|\d+\.\d+|\d+|[A-Z][A-Z ]*[A-Z]')
    matches = pattern.findall(libro_str)
    result = []
    for match in matches:
        if re.match(r'\d+\.\d+', match):
            result.append(float(match))
        elif re.match(r'\d+', match):
            result.append(int(match))
        else:
            result.append(match.strip())

    libro_limpio = [limpiar_texto(str(item)) for item in result if limpiar_texto(str(item))]
    libro_usuario = [item.upper() for item in libro_limpio]

    # Carga del dataset
    df = pd.read_csv('grupo13_pp2\data\Datos_Integrados.csv', header=0, encoding='latin-1')
    titles = [title.upper().strip() for title in df['titulo'].tolist()]

    # Encontrar coincidencias
    coincidencias = encontrar_coincidencias(libro_usuario, titles, umbral=0.75)

    if coincidencias:
        

        coincidencia_index = max(coincidencias, key=lambda tupla: len(max(tupla, key=len)))
        coincidencia_mas_relevante = coincidencia_index[1]

        st.write("Se encontraron las siguientes coincidencias:")
        df_resultado = df[df['titulo'].str.upper().str.strip() == coincidencia_mas_relevante]
        df_resultado = df_resultado.drop(columns=['portada'])
        df_resultado = df_resultado.reset_index(drop=True)
        df_resultado.columns = [col.upper() for col in df_resultado.columns]

        # Ajustar el ancho de la columna DESCRIPCIÓN
        st.write("<style>div[data-testid='column-description'] {width: 100%;}</style>", unsafe_allow_html=True)
        st.dataframe(df_resultado, height=600)
    else:
        st.warning("No se encontraron coincidencias.")
else:
    st.warning('Por favor, carga una imagen.')


