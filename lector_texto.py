import streamlit as st
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Carga de datos
url = "https://raw.githubusercontent.com/sosarodrigox/grupo13_pp2/main/data/data.csv"
data = pd.read_csv(url)

# Preprocesamiento
# Llenar valores faltantes solo en columnas de tipo objeto (cadenas de texto)
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
    st.write(f"Libro seleccionado por el usuario: {nombre_libro}")
    lista_titulos_completa = data["title"].tolist()
    encontrar_cercanos = difflib.get_close_matches(nombre_libro, lista_titulos_completa)

    if encontrar_cercanos:
        cercanos = encontrar_cercanos[0]
        indice_de_libro = data[data.title == cercanos].index[0]
        puntaje_similitud = list(enumerate(similitud[indice_de_libro]))
        libros_similares_ordenados = sorted(
            puntaje_similitud, key=lambda x: x[1], reverse=True
        )

        # Lista de diccionarios con los libros recomendados y su similitud
        recomendaciones = []
        # Muestra los 10 primeros
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

        # Convertir a un DataFrame
        df_recomendaciones = pd.DataFrame(recomendaciones)
        return df_recomendaciones

    else:
        st.write("No se encontró ninguna coincidencia para el libro ingresado.")
        return pd.DataFrame()  # DataFrame vacío si no hay coincidencias


# Interfaz de usuario
st.title("Bienvenido a la Sugerencia de Libros de la Biblioteca Popular")
st.write(
    "Ingrese el nombre de su libro favorito y buscaremos libros similares para usted."
)

# Entrada de usuario
libro_usuario = st.text_input("Ingrese el nombre de su libro favorito:")

if st.button("Buscar Similares"):
    if libro_usuario:
        df_recomendaciones = recomendar_libros(libro_usuario)
        if not df_recomendaciones.empty:
            st.write("Libros similares encontrados:")
            st.dataframe(df_recomendaciones)
        else:
            st.write("Lo sentimos, no se encontraron libros similares.")
    else:
        st.write("Por favor, ingrese un título de libro.")
