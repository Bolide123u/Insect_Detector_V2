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

    # Onglets pour l'application
    tab1, tab2 = st.tabs(["Application", "Guide d'utilisation"])
    
    with tab1:
        # Charger l'image
        uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Lire le contenu du fichier
            file_bytes = uploaded_file.getvalue()
            
            # Convertir le fichier en image
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Afficher l'image originale
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Image originale", use_column_width=True)

            # Paramètres ajustables
            st.sidebar.header("Paramètres de détection")
            
            # Demander le nombre attendu d'insectes
            expected_insects = st.sidebar.number_input("Nombre d'insectes attendus", min_value=1, value=5, step=1)
            
            presets = {
                "Par défaut": {
                    "blur_kernel": 7, "adapt_block_size": 35, "adapt_c": 5,
                    "morph_kernel": 1, "morph_iterations": 3, "min_area": 1000, "margin": 17
                },
                "Grands insectes": {
                    "blur_kernel": 7, "adapt_block_size": 35, "adapt_c": 8,
                    "morph_kernel": 5, "morph_iterations": 2, "min_area": 300, "margin": 15
                },
                "Petits insectes": {
                    "blur_kernel": 3, "adapt_block_size": 15, "adapt_c": 2,
                    "morph_kernel": 3, "morph_iterations": 1, "min_area": 30, "margin": 5
                },
                "Haute précision": {
                    "blur_kernel": 5, "adapt_block_size": 25, "adapt_c": 12,
                    "morph_kernel": 5, "morph_iterations": 3, "min_area": 150, "margin": 10
                }
            }
            
            preset_choice = st.sidebar.selectbox(
                "Configurations prédéfinies", 
                ["Personnalisé", "Auto-ajustement"] + list(presets.keys()),
                index=2
            )
            
            blur_kernel, adapt_block_size, adapt_c, morph_kernel, morph_iterations, min_area, margin = 7, 35, 5, 1, 3, 1000, 17
            auto_adjust = False
            
            if preset_choice == "Auto-ajustement":
                st.sidebar.info(f"Les paramètres seront ajustés automatiquement pour détecter {expected_insects} insectes.")
                auto_adjust = True
                blur_kernel = st.sidebar.slider("Taille du noyau de flou gaussien", 1, 21, 7, step=2)
                adapt_block_size = st.sidebar.slider("Taille du bloc adaptatif", 3, 51, 35, step=2)
                morph_kernel = st.sidebar.slider("Taille du noyau morphologique", 1, 9, 1, step=2)
                morph_iterations = st.sidebar.slider("Itérations morphologiques", 1, 5, 3)
            elif preset_choice != "Personnalisé":
                preset = presets[preset_choice]
                blur_kernel = st.sidebar.slider("Taille du noyau de flou gaussien", 1, 21, preset["blur_kernel"], step=2)
                adapt_block_size = st.sidebar.slider("Taille du bloc adaptatif", 3, 51, preset["adapt_block_size"], step=2)
                adapt_c = st.sidebar.slider("Constante de seuillage adaptatif", -10, 30, preset["adapt_c"])
                morph_kernel = st.sidebar.slider("Taille du noyau morphologique", 1, 9, preset["morph_kernel"], step=2)
                morph_iterations = st.sidebar.slider("Itérations morphologiques", 1, 5, preset["morph_iterations"])
                min_area = st.sidebar.slider("Surface minimale (pixels)", 10, 1000, preset["min_area"])
                margin = st.sidebar.slider("Marge autour des insectes", 0, 50, preset["margin"])
            else:
                blur_kernel = st.sidebar.slider("Taille du noyau de flou gaussien", 1, 21, 7, step=2)
                adapt_block_size = st.sidebar.slider("Taille du bloc adaptatif", 3, 51, 35, step=2)
                adapt_c = st.sidebar.slider("Constante de seuillage adaptatif", -10, 30, 5)
                morph_kernel = st.sidebar.slider("Taille du noyau morphologique", 1, 9, 1, step=2)
                morph_iterations = st.sidebar.slider("Itérations morphologiques", 1, 5, 3)
                min_area = st.sidebar.slider("Surface minimale (pixels)", 10, 1000, 1000)
                margin = st.sidebar.slider("Marge autour des insectes", 0, 50, 17)

            use_circularity = st.sidebar.checkbox("Filtrer par circularité", value=False)
            if use_circularity:
                min_circularity = st.sidebar.slider("Circularité minimale", 0.0, 1.0, 0.3)

            with st.spinner("Traitement de l'image en cours..."):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if blur_kernel > 1: blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
                else: blurred = gray
                
                filtered_props = [] # Initialisation

                if auto_adjust:
                    adapt_c_values = [-5, 0, 2, 5, 8, 10, 15]
                    min_area_values = [20, 30, 50, 75, 100, 150, 200, 300]
                    st.info("Recherche des meilleurs paramètres en cours...")
                    total_iterations = len(adapt_c_values) * len(min_area_values)
                    progress_bar = st.progress(0)
                    iteration_counter = 0
                    best_params = {"adapt_c": 5, "min_area": 50}
                    best_count_diff = float('inf')
                    
                    for ac_test in adapt_c_values:
                        for ma_test in min_area_values:
                            iteration_counter += 1
                            progress_bar.progress(iteration_counter / total_iterations)
                            thresh_test = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adapt_block_size, ac_test)
                            kernel_test = np.ones((morph_kernel, morph_kernel), np.uint8)
                            opening_test = cv2.morphologyEx(thresh_test, cv2.MORPH_OPEN, kernel_test, iterations=morph_iterations)
                            cleared_test = clear_border(opening_test)
                            labels_test = measure.label(cleared_test)
                            props_test = measure.regionprops(labels_test)
                            current_filtered_props_test = [prop for prop in props_test if prop.area >= ma_test]
                            count_diff = abs(len(current_filtered_props_test) - expected_insects)
                            if count_diff < best_count_diff:
                                best_count_diff = count_diff
                                best_params["adapt_c"] = ac_test
                                best_params["min_area"] = ma_test
                                filtered_props = current_filtered_props_test # Mettre à jour filtered_props ici
                    
                    adapt_c = best_params["adapt_c"]
                    min_area = best_params["min_area"]
                    st.success(f"Paramètres optimaux trouvés: adapt_c={adapt_c}, min_area={min_area}")
                    
                    # Recalculer une dernière fois avec les meilleurs paramètres pour 'thresh' et 'opening' pour affichage
                    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adapt_block_size, adapt_c)
                    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
                    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
                    cleared = clear_border(opening) # important pour la cohérence
                    labels = measure.label(cleared) # recalculer labels aussi
                    # filtered_props est déjà défini avec les meilleurs props

                else: # Traitement standard
                    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adapt_block_size, adapt_c)
                    connect_kernel = np.ones((5, 5), np.uint8)
                    dilated_thresh = cv2.dilate(thresh, connect_kernel, iterations=2)
                    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
                    closing = cv2.morphologyEx(dilated_thresh, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
                    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=1)
                    cleared = clear_border(opening)
                    labels = measure.label(cleared)
                    props = measure.regionprops(labels)
                    
                    if use_circularity:
                        filtered_props = []
                        for prop in props:
                            if prop.area >= min_area:
                                perimeter = prop.perimeter
                                if perimeter > 0:
                                    circularity = 4 * np.pi * prop.area / (perimeter * perimeter)
                                    if circularity >= min_circularity:
                                        filtered_props.append(prop)
                    else:
                        filtered_props = [prop for prop in props if prop.area >= min_area]

                col_vis1, col_vis2 = st.columns(2)
                with col_vis1:
                    st.image(blurred, caption="Image floutée", use_column_width=True)
                    st.image(thresh, caption="Après seuillage adaptatif", use_column_width=True)
                with col_vis2:
                    st.image(opening, caption="Après opérations morphologiques", use_column_width=True)
                    label_display = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
                    for i, prop in enumerate(filtered_props): # Utiliser filtered_props pour l'affichage
                        color = np.random.randint(0, 255, size=3)
                        for coord in prop.coords:
                            label_display[coord[0], coord[1]] = color
                    st.image(label_display, caption=f"Insectes détectés: {len(filtered_props)}", use_column_width=True)

                st.subheader("Statistiques de détection")
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                stat_col1.metric("Nombre d'insectes", len(filtered_props))
                stat_col1.metric("Nombre attendu", expected_insects)
                if filtered_props:
                    areas = [prop.area for prop in filtered_props]
                    stat_col2.metric("Surface moyenne (px)", f"{int(np.mean(areas))}")
                    stat_col3.metric("Plage de tailles (px)", f"{int(min(areas))} - {int(max(areas))}")
                
                diff = abs(len(filtered_props) - expected_insects)
                if diff == 0: st.success(f"✅ Nombre exact d'insectes détectés: {len(filtered_props)}")
                elif diff <= 2: st.warning(f"⚠️ {len(filtered_props)} insectes détectés (écart de {diff} par rapport au nombre attendu)")
                else:
                    st.error(f"❌ {len(filtered_props)} insectes détectés (écart important de {diff} par rapport au nombre attendu)")
                    if not auto_adjust and st.button("Essayer l'auto-ajustement"):
                        st.session_state['auto_adjust'] = True
                        st.session_state['preset_choice'] = "Auto-ajustement"
                        st.experimental_rerun()

                if st.button("Extraire et télécharger les insectes isolés (standardisés carrés)"):
                    temp_dir = tempfile.mkdtemp()
                    zip_path = os.path.join(temp_dir, "insectes_isoles_carres.zip")
                
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for i, prop in enumerate(filtered_props):
                            minr, minc, maxr, maxc = prop.bbox
                            minr = max(0, minr - margin)
                            minc = max(0, minc - margin)
                            maxr = min(image.shape[0], maxr + margin)
                            maxc = min(image.shape[1], maxc + margin)
                
                            insect_roi = image[minr:maxr, minc:maxc].copy()
                            roi_height, roi_width = insect_roi.shape[:2]
                            
                            if roi_height == 0 or roi_width == 0: # Skip if ROI is empty
                                continue

                            mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
                            for coord in prop.coords:
                                if minr <= coord[0] < maxr and minc <= coord[1] < maxc:
                                    mask[coord[0] - minr, coord[1] - minc] = 255
                            
                            kernel_close = np.ones((7, 7), np.uint8)
                            mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=5)
                            contours, _ = cv2.findContours(mask_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                            filled_mask = np.zeros_like(mask)
                            for contour in contours:
                                if cv2.contourArea(contour) > 20:
                                    cv2.drawContours(filled_mask, [contour], -1, 255, thickness=cv2.FILLED)
                            kernel_dilate = np.ones((5, 5), np.uint8)
                            dilated_mask = cv2.dilate(filled_mask, kernel_dilate, iterations=3)
                            final_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel_close, iterations=4)
                            mask_with_border = cv2.copyMakeBorder(final_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
                            flood_mask = np.zeros((roi_height+4, roi_width+4), dtype=np.uint8)
                            cv2.floodFill(mask_with_border, flood_mask, (0, 0), 128)
                            holes = np.where((mask_with_border != 128) & (mask_with_border != 255), 255, 0).astype(np.uint8)
                            holes = holes[1:-1, 1:-1]
                            complete_mask = cv2.bitwise_or(final_mask, holes)
                            kernel_smooth = np.ones((3, 3), np.uint8)
                            smooth_mask = cv2.dilate(complete_mask, kernel_smooth, iterations=1)
                            
                            # Image avec fond blanc (rectangulaire originale)
                            mask_3ch = cv2.cvtColor(smooth_mask, cv2.COLOR_GRAY2BGR)
                            white_bg_roi = np.ones_like(insect_roi) * 255
                            insect_on_white_roi = np.where(mask_3ch == 255, insect_roi, white_bg_roi)
                            
                            # Image avec transparence (rectangulaire originale)
                            insect_transparent_roi = np.zeros((roi_height, roi_width, 4), dtype=np.uint8)
                            insect_transparent_roi[:, :, :3] = insect_roi
                            insect_transparent_roi[:, :, 3] = smooth_mask

                            # --- MODIFICATION POUR IMAGE CARRÉE ---
                            square_dim = max(roi_height, roi_width)
                            
                            # Pour l'image avec fond blanc
                            square_img_white_bg = np.ones((square_dim, square_dim, 3), dtype=np.uint8) * 255
                            # Calculer les offsets pour centrer
                            y_offset = (square_dim - roi_height) // 2
                            x_offset = (square_dim - roi_width) // 2
                            # Placer l'insecte sur fond blanc (ROI) au centre de l'image carrée
                            square_img_white_bg[y_offset:y_offset+roi_height, x_offset:x_offset+roi_width] = insect_on_white_roi
                            
                            temp_img_path_square_white = os.path.join(temp_dir, f"insect_{i+1}_square_white.jpg")
                            cv2.imwrite(temp_img_path_square_white, square_img_white_bg)
                            zipf.write(temp_img_path_square_white, f"insect_{i+1}_square_white.jpg")

                            # Pour l'image avec fond transparent
                            square_img_transparent_bg = np.zeros((square_dim, square_dim, 4), dtype=np.uint8) # Fond transparent par défaut (alpha=0)
                            # Placer l'insecte avec transparence (ROI) au centre
                            square_img_transparent_bg[y_offset:y_offset+roi_height, x_offset:x_offset+roi_width] = insect_transparent_roi

                            temp_img_path_square_transparent = os.path.join(temp_dir, f"insect_{i+1}_square_transparent.png")
                            cv2.imwrite(temp_img_path_square_transparent, square_img_transparent_bg)
                            zipf.write(temp_img_path_square_transparent, f"insect_{i+1}_square_transparent.png")
                            # --- FIN MODIFICATION ---
                
                    with open(zip_path, "rb") as f:
                        bytes_data = f.read()
                        b64 = base64.b64encode(bytes_data).decode()
                        href = f'<a href="data:application/zip;base64,{b64}" download="insectes_isoles_carres.zip">Télécharger tous les insectes isolés (ZIP, images carrées)</a>'
                        st.markdown(href, unsafe_allow_html=True)
                
                    if filtered_props:
                        st.write("Aperçu des premiers insectes isolés (standardisés carrés):")
                        preview_cols = st.columns(min(5, len(filtered_props)))
                
                        for i, col in enumerate(preview_cols):
                            if i < len(filtered_props):
                                prop = filtered_props[i]
                                minr, minc, maxr, maxc = prop.bbox
                                minr = max(0, minr - margin)
                                minc = max(0, minc - margin)
                                maxr = min(image.shape[0], maxr + margin)
                                maxc = min(image.shape[1], maxc + margin)
                
                                insect_roi = image[minr:maxr, minc:maxc].copy()
                                roi_height, roi_width = insect_roi.shape[:2]

                                if roi_height == 0 or roi_width == 0: continue
                                
                                mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
                                for coord in prop.coords:
                                    if minr <= coord[0] < maxr and minc <= coord[1] < maxc:
                                        mask[coord[0] - minr, coord[1] - minc] = 255
                                kernel_close = np.ones((7, 7), np.uint8)
                                mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=5)
                                contours, _ = cv2.findContours(mask_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                                filled_mask = np.zeros_like(mask)
                                for contour in contours:
                                    if cv2.contourArea(contour) > 20:
                                        cv2.drawContours(filled_mask, [contour], -1, 255, thickness=cv2.FILLED)
                                kernel_dilate = np.ones((5, 5), np.uint8)
                                dilated_mask = cv2.dilate(filled_mask, kernel_dilate, iterations=3)
                                final_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel_close, iterations=4)
                                mask_with_border = cv2.copyMakeBorder(final_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
                                flood_mask = np.zeros((roi_height+4, roi_width+4), dtype=np.uint8)
                                cv2.floodFill(mask_with_border, flood_mask, (0, 0), 128)
                                holes = np.where((mask_with_border != 128) & (mask_with_border != 255), 255, 0).astype(np.uint8)
                                holes = holes[1:-1, 1:-1]
                                complete_mask = cv2.bitwise_or(final_mask, holes)
                                kernel_smooth = np.ones((3, 3), np.uint8)
                                smooth_mask = cv2.dilate(complete_mask, kernel_smooth, iterations=1)
                                
                                mask_3ch = cv2.cvtColor(smooth_mask, cv2.COLOR_GRAY2BGR)
                                white_bg_roi = np.ones_like(insect_roi) * 255
                                insect_on_white_roi = np.where(mask_3ch == 255, insect_roi, white_bg_roi)

                                # --- MODIFICATION POUR APERÇU CARRÉ ---
                                square_dim_preview = max(roi_height, roi_width)
                                square_preview_img_white_bg = np.ones((square_dim_preview, square_dim_preview, 3), dtype=np.uint8) * 255
                                y_offset_preview = (square_dim_preview - roi_height) // 2
                                x_offset_preview = (square_dim_preview - roi_width) // 2
                                square_preview_img_white_bg[y_offset_preview:y_offset_preview+roi_height, x_offset_preview:x_offset_preview+roi_width] = insect_on_white_roi
                                # --- FIN MODIFICATION APERÇU ---

                                col.image(cv2.cvtColor(square_preview_img_white_bg, cv2.COLOR_BGR2RGB), caption=f"Insecte {i+1}", use_column_width=True)
    
    with tab2:
        st.header("Guide d'optimisation des paramètres")
        st.subheader("Configurations prédéfinies")
        st.write("""
        L'application propose plusieurs configurations prédéfinies pour différents types d'images:
        - **Par défaut**: Configuration optimisée (flou: 7, bloc adaptatif: 35, seuillage: 5, noyau morpho: 1, itérations: 3, surface min: 1000, marge: 17)
        - **Grands insectes**: Optimisée pour détecter des insectes de grande taille
        - **Petits insectes**: Optimisée pour les insectes de petite taille ou les détails fins
        - **Haute précision**: Réduit les fausses détections au prix d'une sensibilité légèrement plus faible
        - **Arthropodes à pattes fines**: Conçue pour les insectes avec des appendices fins qui pourraient être perdus.
        - **Auto-ajustement**: Ajuste automatiquement les paramètres pour détecter le nombre d'insectes spécifié.
        
        Vous pouvez commencer avec l'une de ces configurations puis ajuster les paramètres selon vos besoins.
        """)
        st.subheader("Utilisation de l'auto-ajustement")
        st.write("""
        La fonctionnalité d'auto-ajustement permet de spécifier le nombre d'insectes attendus dans l'image:
        1. Indiquez le nombre d'insectes que vous savez présents dans l'image.
        2. Sélectionnez le mode "Auto-ajustement" dans les configurations prédéfinies.
        3. L'application testera différentes combinaisons de paramètres pour trouver celle qui détecte au mieux le nombre souhaité.
        Cette approche est particulièrement utile lorsque vous connaissez le nombre exact d'insectes.
        """)
        st.subheader("Problèmes courants et solutions")
        col1_guide, col2_guide = st.columns(2)
        with col1_guide:
            st.markdown("#### Trop d'insectes détectés")
            st.write("""
            1. **Augmentez la surface minimale** (min_area).
            2. **Augmentez la constante de seuillage** (adapt_c).
            3. **Activez le filtrage par circularité** et ajustez le seuil.
            4. **Augmentez la taille du noyau morphologique** et/ou le nombre d'itérations.
            """)
        with col2_guide:
            st.markdown("#### Insectes manquants ou incomplets")
            st.write("""
            1. **Diminuez la constante de seuillage** (adapt_c).
            2. **Diminuez la surface minimale** (min_area).
            3. **Ajustez la taille du bloc adaptatif**.
            4. **Réduisez le noyau de flou gaussien**.
            """)
        st.subheader("Guide étape par étape")
        st.write("""
        1. **Commencez avec une configuration prédéfinie**.
        2. **Affinez le seuillage adaptatif**: `adapt_block_size` (plus grand que l'insecte), `adapt_c` (sensibilité).
        3. **Optimisez les opérations morphologiques**: `morph_kernel` et `morph_iterations`.
        4. **Ajustez le filtrage**: `min_area`, circularité.
        5. **Vérifiez visuellement** et affinez.
        """)
        st.markdown("### Exemples de paramètres efficaces")
        param_examples = [
            {"type": "Insectes bien contrastés", "blur": "3-5", "block": "15-25", "c": "5-10", "morph": "3, 1-2 it.", "area": "100-200"},
            {"type": "Insectes à faible contraste", "blur": "3", "block": "15-21", "c": "2-5", "morph": "3, 1 it.", "area": "50-100"},
            {"type": "Grands insectes avec détails", "blur": "5-7", "block": "31-41", "c": "8-12", "morph": "5, 2-3 it.", "area": "300-500"}
        ]
        table_data = {
            "Type d'image": [ex["type"] for ex in param_examples], "Flou": [ex["blur"] for ex in param_examples],
            "Bloc adaptatif": [ex["block"] for ex in param_examples], "Constante C": [ex["c"] for ex in param_examples],
            "Morphologie": [ex["morph"] for ex in param_examples], "Surface min.": [ex["area"] for ex in param_examples],
        }
        st.table(table_data)
        st.info("Astuce: Prétraitez vos images pour améliorer le contraste avant de les charger.")

if __name__ == "__main__":
    main()
