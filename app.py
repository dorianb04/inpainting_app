from io import BytesIO
import streamlit as st
from utils.model import configure_model
from torchvision import transforms
import torch
import cv2
from PIL import Image, ImageDraw
from PIL import ImageOps
import torchvision.transforms as T
from streamlit_drawable_canvas import st_canvas

###############################
#### Déclaration des variables
###############################
st.set_page_config(layout="centered")

generator, discriminator = configure_model()

transform_input_image = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


def hex_to_rgb(hex):
    return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))


###############################
#### Affichage de l'application
###############################
st.title("Reconstruisez vos photos avec un modèle d'inpainting")
st.subheader("Réalisé par Dorian Boucher dans le cadre d'un PE encadré par l'UTT")
bg_image = st.file_uploader("Selectionnez votre image à transformer :", type=["png", "jpg"])

if bg_image:
    img_name = bg_image.name
    bg_image = ImageOps.exif_transpose(Image.open(bg_image))
    with st.sidebar:
        drawing_mode = st.selectbox(
            "Type de dessin:",
            ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
        )
        stroke_width = st.slider("Epaisseur de la mine : ", 1, 25, 3)
        if drawing_mode == 'point':
            point_display_radius = st.slider("Rayon du point : ", 1, 25, 3)
        stroke_color = hex_to_rgb(f'{st.color_picker("Couleur de la mine : ")[1:]}')
        bg_color = "#eee"
        realtime_update = st.checkbox("Transformation en temps réel", True)

        padding_data = torch.load('utils/padding_data.pt')

    st.markdown("## Selectionnez la partie de l'image que vous souhaitez éditer")

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=f"rgba({stroke_color[0]}, {stroke_color[1]}, {stroke_color[2]}, 0.5)",
        background_color=bg_color,
        background_image=bg_image,
        update_streamlit=realtime_update,
        width=500,
        height=500,
        drawing_mode=drawing_mode,
        #point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        display_toolbar=st.sidebar.checkbox("Affichage des outils d'édition de l'image", True),
        key="full_app",
    )

    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        if bg_image:
            # Tranformation du masque et de l'image obtenue

            # Transformation de l'image de base en tenseur pytorch 64x64
            img_base = transform_input_image(bg_image)[:3]

            # Transformation du masque obtenu en matrice 64x64 normalisée entre 0 et 1
            mask = cv2.resize((255 - canvas_result.image_data[:, :, 3]) / 255, (64, 64))

            # Transformation de la matrice du masque avec des 1 pour les pixels non masqués et des 0 pour les pixels masqués
            mask[mask != 1] = 0

            # Application du masque à l'image non masquée
            img_masked = img_base * mask

            # Transformation du mask pour le convertir au bon format du model
            mask = torch.tensor(mask).repeat(3, 1, 1)

            # Concatenation de l'image masquée et du masque avec des données entrainées pour conditionner le modèle
            img_masked_combined = torch.cat((img_masked.unsqueeze(0), padding_data[0][1:])).float()
            mask_combined = torch.cat((mask.unsqueeze(0), padding_data[1][1:])).float()

            # Application du générateur
            preds = generator(img_masked_combined, mask_combined)[0]

            # Affichage des résultats
            st.markdown("### Votre image restaurée par le modèle d'inpainting")
            st.markdown("###### (Limité pour le moment en qualité 64x64)")
            st.image(T.ToPILImage()(preds), width=500)

            # Création du bouton de téléchargement de l'image générée
            buf = BytesIO()
            T.ToPILImage()(preds).save(buf, format="JPEG")
            byte_im = buf.getvalue()

            btn = st.download_button(
                label="Téléchargez votre nouvelle image",
                data=byte_im,
                file_name=f"inpainted_{img_name}.png",
                mime="image/jpeg",
            )
