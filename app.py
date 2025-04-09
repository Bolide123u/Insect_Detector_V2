# Installer les bibliothèques nécessaires
!pip install streamlit opencv-python matplotlib scikit-image pillow pyngrok

# Importer le module streamlit pour vérifier qu'il est bien installé
import streamlit

# Créer un fichier temporaire pour l'application Streamlit
%%writefile app.py
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.segmentation import clear_border
import os
import io
from PIL import Image
import tempfile
import zipfile
import base64

def main():
    st.title("Détection et isolation d'insectes")
    st.write("Cette application permet de détecter des insectes sur un fond clair et de les isoler individuellement.")

    # Charger l'image
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convertir le fichier en image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Afficher l'image originale
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Image originale", use_column_width=True)

        # Paramètres ajustables
        st.sidebar.header("Paramètres de détection")

        blur_kernel = st.sidebar.slider("Taille du noyau de flou gaussien", 1, 21, 5, step=2)
        adapt_block_size = st.sidebar.slider("Taille du bloc adaptatif", 3, 51, 21, step=2)
        adapt_c = st.sidebar.slider("Constante de seuillage adaptatif", -10, 30, 5)

        morph_kernel = st.sidebar.slider("Taille du noyau morphologique", 1, 9, 3, step=2)
        morph_iterations = st.sidebar.slider("Itérations morphologiques", 1, 5, 1)

        min_area = st.sidebar.slider("Surface minimale (pixels)", 10, 1000, 50)
        margin = st.sidebar.slider("Marge autour des insectes", 0, 50, 10)

        # Traitement de l'image
        with st.spinner("Traitement de l'image en cours..."):
            # Convertir l'image en niveaux de gris
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Appliquer un flou gaussien
            if blur_kernel > 1:
                blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
            else:
                blurred = gray

            # Seuillage adaptatif
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, adapt_block_size, adapt_c
            )

            # Opérations morphologiques
            kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)

            # Supprimer les objets qui touchent les bords
            cleared = clear_border(opening)

            # Étiqueter les composants connectés
            labels = measure.label(cleared)

            # Obtenir les propriétés des régions
            props = measure.regionprops(labels)

            # Filtrer les petites régions
            filtered_props = [prop for prop in props if prop.area >= min_area]

            # Créer une visualisation des étapes
            col1, col2 = st.columns(2)

            with col1:
                st.image(blurred, caption="Image floutée", use_column_width=True)
                st.image(thresh, caption="Après seuillage adaptatif", use_column_width=True)

            with col2:
                st.image(opening, caption="Après opérations morphologiques", use_column_width=True)

                # Créer une image colorée des labels pour visualisation
                label_display = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
                for i, prop in enumerate(filtered_props):
                    color = np.random.randint(0, 255, size=3)
                    for coord in prop.coords:
                        label_display[coord[0], coord[1]] = color

                st.image(label_display, caption=f"Insectes détectés: {len(filtered_props)}", use_column_width=True)

            st.success(f"{len(filtered_props)} insectes ont été détectés!")

            # Option pour extraire et télécharger les insectes
            if st.button("Extraire et télécharger les insectes isolés"):
                # Créer un dossier temporaire
                temp_dir = tempfile.mkdtemp()
                zip_path = os.path.join(temp_dir, "insectes_isoles.zip")

                # Créer un fichier zip
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    # Pour chaque insecte détecté
                    for i, prop in enumerate(filtered_props):
                        # Obtenir les coordonnées de la boîte englobante
                        minr, minc, maxr, maxc = prop.bbox

                        # Ajouter une marge
                        minr = max(0, minr - margin)
                        minc = max(0, minc - margin)
                        maxr = min(image.shape[0], maxr + margin)
                        maxc = min(image.shape[1], maxc + margin)

                        # Extraire l'insecte avec sa boîte englobante
                        insect_roi = image[minr:maxr, minc:maxc]

                        # Créer un masque pour isoler l'insecte du fond
                        mask = np.zeros_like(gray)
                        for coords in prop.coords:
                            mask[coords[0], coords[1]] = 255

                        # Extraire le masque pour cet insecte spécifique
                        insect_mask = mask[minr:maxr, minc:maxc]

                        # Créer une image avec fond blanc
                        white_bg = np.ones_like(insect_roi) * 255

                        # Créer un masque binaire de l'insecte
                        _, binary_mask = cv2.threshold(insect_mask, 127, 255, cv2.THRESH_BINARY)
                        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

                        # Utiliser le masque pour combiner l'insecte avec le fond blanc
                        insect_on_white = np.where(binary_mask == 255, insect_roi, white_bg)

                        # Sauvegarder l'image temporairement
                        temp_img_path = os.path.join(temp_dir, f"insect_{i+1}.png")
                        cv2.imwrite(temp_img_path, insect_on_white)

                        # Ajouter au zip
                        zipf.write(temp_img_path, f"insect_{i+1}.png")

                # Créer un lien de téléchargement pour le zip
                with open(zip_path, "rb") as f:
                    bytes_data = f.read()
                    b64 = base64.b64encode(bytes_data).decode()
                    href = f'<a href="data:application/zip;base64,{b64}" download="insectes_isoles.zip">Télécharger tous les insectes isolés (ZIP)</a>'
                    st.markdown(href, unsafe_allow_html=True)

                # Afficher quelques exemples d'insectes isolés (les 5 premiers)
                if filtered_props:
                    st.write("Aperçu des premiers insectes isolés:")
                    preview_cols = st.columns(min(5, len(filtered_props)))

                    for i, col in enumerate(preview_cols):
                        if i < len(filtered_props):
                            prop = filtered_props[i]
                            minr, minc, maxr, maxc = prop.bbox
                            minr = max(0, minr - margin)
                            minc = max(0, minc - margin)
                            maxr = min(image.shape[0], maxr + margin)
                            maxc = min(image.shape[1], maxc + margin)

                            insect_roi = image[minr:maxr, minc:maxc]
                            mask = np.zeros_like(gray)
                            for coords in prop.coords:
                                mask[coords[0], coords[1]] = 255
                            insect_mask = mask[minr:maxr, minc:maxc]
                            white_bg = np.ones_like(insect_roi) * 255
                            _, binary_mask = cv2.threshold(insect_mask, 127, 255, cv2.THRESH_BINARY)
                            binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
                            insect_on_white = np.where(binary_mask == 255, insect_roi, white_bg)

                            col.image(cv2.cvtColor(insect_on_white, cv2.COLOR_BGR2RGB), caption=f"Insecte {i+1}", use_column_width=True)

    # Instructions d'utilisation
    else:
        st.info("📋 Instructions:")
        st.write("""
        1. Téléchargez une image contenant des insectes sur fond clair
        2. Ajustez les paramètres dans la barre latérale pour optimiser la détection
        3. Visualisez les résultats en temps réel
        4. Extrayez et téléchargez les insectes isolés lorsque vous êtes satisfait des résultats
        """)

        st.write("""
        ### Paramètres expliqués:
        - **Taille du noyau de flou gaussien**: contrôle l'intensité du flou appliqué pour réduire le bruit
        - **Taille du bloc adaptatif**: taille de la zone utilisée pour calculer le seuil local (valeurs élevées pour les grands insectes)
        - **Constante de seuillage adaptatif**: ajuste la sensibilité du seuillage (valeurs négatives = plus sensible)
        - **Taille du noyau morphologique**: taille du filtre pour les opérations morphologiques
        - **Itérations morphologiques**: nombre d'applications successives des opérations morphologiques
        - **Surface minimale**: taille minimale en pixels pour qu'un objet soit considéré comme un insecte
        - **Marge**: espace supplémentaire autour de chaque insecte isolé
        """)

if __name__ == "__main__":
    main()

# Lancer Streamlit avec ngrok
from pyngrok import ngrok
import subprocess

# Démarrer Streamlit en arrière-plan
!streamlit run app.py &>/dev/null&

# Créer un tunnel avec ngrok
public_url = ngrok.connect(8501)
print(f"L'application Streamlit est accessible à l'adresse: {public_url}")
