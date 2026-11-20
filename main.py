
import streamlit as st
from PIL import Image
from io import BytesIO

from filtros import FILTROS

# Configuracao da pagina
st.set_page_config(
    page_title="ArtFilter",
    page_icon="🎨",
    layout="wide"
)

# Titulo
st.title("🎨 ArtFilter")
st.write(
    "Transforme suas fotografias em diferentes "
    "estilos artísticos!"
)

st.divider()

# Upload da fotografia
arquivo = st.file_uploader(
    "Escolha uma fotografia:",
    type=["jpg", "jpeg", "png"]
)

if arquivo is not None:

    imagem = Image.open(arquivo).convert("RGB")

    st.subheader("Escolha seu filtro")

    filtro_escolhido = st.selectbox(
        "Estilos disponíveis:",
        list(FILTROS.keys())
    )

    # Aplicar filtro
    if st.button("✨ Aplicar filtro"):

        with st.spinner("Criando sua arte..."):

            resultado = FILTROS[filtro_escolhido](imagem)

            st.session_state["resultado"] = resultado
            st.session_state["filtro"] = filtro_escolhido

    # Mostrar imagens
    if "resultado" in st.session_state:

        st.divider()

        st.subheader("Resultado")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                imagem,
                caption="Imagem original",
                use_container_width=True
            )

        with col2:
            st.image(
                st.session_state["resultado"],
                caption=st.session_state["filtro"],
                use_container_width=True
            )

        # Preparar download
        buffer = BytesIO()

        st.session_state["resultado"].save(
            buffer, format="PNG"
        )

        st.download_button(
            label="⬇️ Baixar imagem",
            data=buffer.getvalue(),
            file_name="artfilter.png",
            mime="image/png"
        )

else:
    st.info(
        "Envie uma fotografia para começar "
        "a utilizar os filtros!"
    )

st.divider()

st.caption(
    "ArtFilter | Projeto acadêmico desenvolvido em Python"
)

