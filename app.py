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
            
            # Ajout de configurations prédéfinies
            presets = {
                "Par défaut": {
                    "blur_kernel": 7,
                    "adapt_block_size": 35,
                    "adapt_c": 5,
                    "morph_kernel": 1,
                    "morph_iterations": 3,
                    "min_area": 1000,  # Valeur maximale comme demandé
                    "margin": 17
                },
                "Grands insectes": {
                    "blur_kernel": 7,
                    "adapt_block_size": 35,
                    "adapt_c": 8,
                    "morph_kernel": 5,
                    "morph_iterations": 2,
                    "min_area": 300,
                    "margin": 15
                },
                "Petits insectes": {
                    "blur_kernel": 3,
                    "adapt_block_size": 15,
                    "adapt_c": 2,
                    "morph_kernel": 3,
                    "morph_iterations": 1,
                    "min_area": 30,
                    "margin": 5
                },
                "Haute précision": {
                    "blur_kernel": 5,
                    "adapt_block_size": 25,
                    "adapt_c": 12,
                    "morph_kernel": 5,
                    "morph_iterations": 3,
                    "min_area": 150,
                    "margin": 10
                }
            }
            
            preset_choice = st.sidebar.selectbox(
                "Configurations prédéfinies", 
                ["Personnalisé", "Auto-ajustement"] + list(presets.keys()),
                index=2  # "Par défaut" sélectionné par défaut (index 2 après "Personnalisé" et "Auto-ajustement")
            )
            
            # Initialisation des paramètres par défaut avec vos nouvelles valeurs optimales
            blur_kernel = 7
            adapt_block_size = 35
            adapt_c = 5
            morph_kernel = 1
            morph_iterations = 3
            min_area = 1000  # Valeur maximale comme demandé
            margin = 17
            auto_adjust = False
            
            # Utiliser les valeurs des presets ou permettre l'ajustement manuel
            if preset_choice == "Auto-ajustement":
                st.sidebar.info(f"Les paramètres seront ajustés automatiquement pour détecter {expected_insects} insectes.")
                auto_adjust = True
                
                # Permettre d'ajuster certains paramètres de base même en mode auto-ajustement
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
                # Paramètres complètement personnalisables - initialisés avec vos valeurs optimales
                blur_kernel = st.sidebar.slider("Taille du noyau de flou gaussien", 1, 21, 7, step=2)
                adapt_block_size = st.sidebar.slider("Taille du bloc adaptatif", 3, 51, 35, step=2)
                adapt_c = st.sidebar.slider("Constante de seuillage adaptatif", -10, 30, 5)
                morph_kernel = st.sidebar.slider("Taille du noyau morphologique", 1, 9, 1, step=2)
                morph_iterations = st.sidebar.slider("Itérations morphologiques", 1, 5, 3)
                min_area = st.sidebar.slider("Surface minimale (pixels)", 10, 1000, 1000)
                margin = st.sidebar.slider("Marge autour des insectes", 0, 50, 17)

            # Ajouter un filtre de circularité
            use_circularity = st.sidebar.checkbox("Filtrer par circularité", value=False)
            if use_circularity:
                min_circularity = st.sidebar.slider("Circularité minimale", 0.0, 1.0, 0.3)

            # Traitement de l'image
            with st.spinner("Traitement de l'image en cours..."):
                # Convertir l'image en niveaux de gris
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Appliquer un flou gaussien
                if blur_kernel > 1:
                    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
                else:
                    blurred = gray
                
                # Si auto-ajustement est activé
                if auto_adjust:
                    # Plages de paramètres à explorer
                    adapt_c_values = [-5, 0, 2, 5, 8, 10, 15]
                    min_area_values = [20, 30, 50, 75, 100, 150, 200, 300]
                    
                    st.info("Recherche des meilleurs paramètres en cours...")
                    
                    # Créer un tableau de progression
                    total_iterations = len(adapt_c_values) * len(min_area_values)
                    progress_bar = st.progress(0)
                    iteration_counter = 0
                    
                    # Variables pour stocker les meilleurs paramètres
                    best_params = {"adapt_c": 5, "min_area": 50}
                    best_count_diff = float('inf')
                    best_filtered_props = []
                    
                    # Tester toutes les combinaisons de paramètres
                    for ac in adapt_c_values:
                        for ma in min_area_values:
                            # Mettre à jour la barre de progression
                            iteration_counter += 1
                            progress_bar.progress(iteration_counter / total_iterations)
                            
                            # Appliquer le seuillage avec les paramètres actuels
                            thresh = cv2.adaptiveThreshold(
                                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, adapt_block_size, ac
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
                            current_filtered_props = [prop for prop in props if prop.area >= ma]
                            
                            # Calculer la différence avec le nombre attendu
                            count_diff = abs(len(current_filtered_props) - expected_insects)
                            
                            # Si cette combinaison donne un résultat plus proche du nombre attendu
                            if count_diff < best_count_diff:
                                best_count_diff = count_diff
                                best_params["adapt_c"] = ac
                                best_params["min_area"] = ma
                                best_filtered_props = current_filtered_props
                    
                    # Utiliser les meilleurs paramètres trouvés
                    adapt_c = best_params["adapt_c"]
                    min_area = best_params["min_area"]
                    
                    # Recalculer une dernière fois avec les meilleurs paramètres
                    thresh = cv2.adaptiveThreshold(
                        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY_INV, adapt_block_size, adapt_c
                    )
                    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
                    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
                    cleared = clear_border(opening)
                    labels = measure.label(cleared)
                    
                    # Afficher les paramètres optimaux trouvés
                    st.success(f"Paramètres optimaux trouvés: adapt_c={adapt_c}, min_area={min_area}")
                    
                    # Utiliser les propriétés filtrées optimales
                    filtered_props = best_filtered_props
                    
                else:
                    # Traitement standard avec les paramètres choisis
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

                    # Filtrer les petites régions et appliquer le filtre de circularité si activé
                    if use_circularity:
                        filtered_props = []
                        for prop in props:
                            if prop.area >= min_area:
                                # Calculer la circularité: 4π × aire / périmètre²
                                perimeter = prop.perimeter
                                if perimeter > 0:  # Éviter division par zéro
                                    circularity = 4 * np.pi * prop.area / (perimeter * perimeter)
                                    if circularity >= min_circularity:
                                        filtered_props.append(prop)
                    else:
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

                # Afficher des statistiques utiles
                st.subheader("Statistiques de détection")
                col1, col2, col3 = st.columns(3)
                col1.metric("Nombre d'insectes", len(filtered_props))
                col1.metric("Nombre attendu", expected_insects)
                
                if filtered_props:
                    areas = [prop.area for prop in filtered_props]
                    col2.metric("Surface moyenne (px)", f"{int(np.mean(areas))}")
                    col3.metric("Plage de tailles (px)", f"{int(min(areas))} - {int(max(areas))}")
                
                # Afficher l'écart par rapport au nombre attendu
                diff = abs(len(filtered_props) - expected_insects)
                if diff == 0:
                    st.success(f"✅ Nombre exact d'insectes détectés: {len(filtered_props)}")
                elif diff <= 2:
                    st.warning(f"⚠️ {len(filtered_props)} insectes détectés (écart de {diff} par rapport au nombre attendu)")
                else:
                    st.error(f"❌ {len(filtered_props)} insectes détectés (écart important de {diff} par rapport au nombre attendu)")
                    
                    # Suggérer l'auto-ajustement si on n'est pas déjà en mode auto
                    if not auto_adjust:
                        if st.button("Essayer l'auto-ajustement"):
                            st.session_state['auto_adjust'] = True
                            st.session_state['preset_choice'] = "Auto-ajustement"
                            st.experimental_rerun()

                # Option pour extraire et télécharger les insectes
                if st.button("Extraire et télécharger les insectes isolés"):
                    # Créer un dossier temporaire
                    temp_dir = tempfile.mkdtemp()
                    zip_path = os.path.join(temp_dir, "insectes_isoles.zip")

                    # Créer un fichier zip
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        # Pour chaque insecte détecté
                        for i, prop in enumerate(filtered_props):
                            # Obtenir les coordonnées de la boîte englobante avec marge
                            minr, minc, maxr, maxc = prop.bbox
                            
                            # Ajouter une marge
                            minr = max(0, minr - margin)
                            minc = max(0, minc - margin)
                            maxr = min(image.shape[0], maxr + margin)
                            maxc = min(image.shape[1], maxc + margin)
                    
                            # Extraire l'insecte avec sa boîte englobante
                            insect_roi = image[minr:maxr, minc:maxc].copy()
                            
                            # Créer un masque initial basé sur les coordonnées détectées
                            mask = np.zeros_like(gray)
                            for coords in prop.coords:
                                mask[coords[0], coords[1]] = 255
                            
                            # Extraire le masque pour cette ROI
                            roi_mask = mask[minr:maxr, minc:maxc]
                            
                            # Préparer le masque pour GrabCut
                            # 0 = fond certain, 1 = fond probable, 2 = objet probable, 3 = objet certain
                            grabcut_mask = np.zeros(roi_mask.shape, dtype=np.uint8)
                            grabcut_mask[roi_mask > 0] = 3  # Marquer l'insecte détecté comme objet certain
                            
                            # Créer un rectangle légèrement plus petit que la ROI pour indiquer la zone d'intérêt
                            rect_margin = max(2, margin // 3)
                            rect = (rect_margin, rect_margin, 
                                    roi_mask.shape[1] - 2*rect_margin, 
                                    roi_mask.shape[0] - 2*rect_margin)
                            
                            # Appliquer GrabCut pour affiner la segmentation
                            bgd_model = np.zeros((1, 65), np.float64)
                            fgd_model = np.zeros((1, 65), np.float64)
                            
                            # Lancer GrabCut avec le masque initial comme guide
                            cv2.grabCut(insect_roi, grabcut_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
                            
                            # Créer le masque final: 0 et 2 sont le fond, 1 et 3 sont l'objet
                            refined_mask = np.where((grabcut_mask==2) | (grabcut_mask==0), 0, 1).astype('uint8')
                            
                            # Convertir en masque 3 canaux
                            refined_mask_3ch = cv2.cvtColor(refined_mask * 255, cv2.COLOR_GRAY2BGR)
                            
                            # Créer un fond blanc
                            white_bg = np.ones_like(insect_roi) * 255
                            
                            # Combiner l'insecte et le fond blanc selon le masque
                            insect_on_white = np.where(refined_mask_3ch > 0, insect_roi, white_bg)
                            
                            # Sauvegarder l'image
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

                    # Afficher quelques exemples d'insectes isolés (limités à 5)
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
                    
                                # Extraire l'insecte avec sa boîte englobante
                                insect_roi = image[minr:maxr, minc:maxc].copy()
                                
                                # Créer un masque pour isoler l'insecte du fond
                                mask = np.zeros_like(gray)
                                for coords in prop.coords:
                                    mask[coords[0], coords[1]] = 255
                                    
                                # Extraire le masque pour cet insecte spécifique
                                insect_mask = mask[minr:maxr, minc:maxc]
                                
                                # Assurer que le masque a la bonne forme (convertir en 3 canaux si nécessaire)
                                if len(insect_mask.shape) == 2:
                                    insect_mask = cv2.cvtColor(insect_mask, cv2.COLOR_GRAY2BGR)
                                
                                # Créer une version dilatée du masque pour adoucir les bords
                                kernel = np.ones((3, 3), np.uint8)
                                dilated_mask = cv2.dilate(insect_mask, kernel, iterations=2)
                                
                                # Créer un fond blanc
                                white_bg = np.ones_like(insect_roi) * 255
                                
                                # Convertir le masque en valeurs flottantes entre 0 et 1
                                alpha_mask = dilated_mask.astype(float) / 255
                                
                                # Appliquer le masque alpha pour une transition plus douce
                                insect_on_white = insect_roi * alpha_mask + white_bg * (1 - alpha_mask)
                                insect_on_white = insect_on_white.astype(np.uint8)
                    
                                col.image(cv2.cvtColor(insect_on_white, cv2.COLOR_BGR2RGB), caption=f"Insecte {i+1}", use_column_width=True)
        else:
            st.info("Veuillez télécharger une image pour commencer.")
    
    # Onglet pour le guide d'utilisation détaillé
    with tab2:
        st.header("Guide d'optimisation des paramètres")
        
        st.subheader("Configurations prédéfinies")
        st.write("""
        L'application propose plusieurs configurations prédéfinies pour différents types d'images:
        - **Par défaut**: Configuration optimisée basée sur les tests (flou gaussien: 7, bloc adaptatif: 35, seuillage: 5, noyau morphologique: 1, itérations: 3, surface min: 1000, marge: 17)
        - **Grands insectes**: Optimisée pour détecter des insectes de grande taille
        - **Petits insectes**: Optimisée pour les insectes de petite taille ou les détails fins
        - **Haute précision**: Réduit les fausses détections au prix d'une sensibilité légèrement plus faible
        - **Auto-ajustement**: Ajuste automatiquement les paramètres pour détecter le nombre d'insectes spécifié
        
        Vous pouvez commencer avec l'une de ces configurations puis ajuster les paramètres selon vos besoins.
        """)
        
        st.subheader("Utilisation de l'auto-ajustement")
        st.write("""
        La fonctionnalité d'auto-ajustement permet de spécifier le nombre d'insectes attendus dans l'image:
        
        1. Indiquez le nombre d'insectes que vous savez présents dans l'image
        2. Sélectionnez le mode "Auto-ajustement" dans les configurations prédéfinies
        3. L'application testera différentes combinaisons de paramètres pour trouver celle qui détecte au mieux le nombre souhaité
        
        Cette approche est particulièrement utile lorsque vous connaissez le nombre exact d'insectes dans l'image et que vous souhaitez optimiser la détection.
        """)
        
        st.subheader("Problèmes courants et solutions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Trop d'insectes détectés")
            st.write("""
            Si l'application détecte trop d'insectes (inclut du bruit ou des artefacts):
            1. **Augmentez la surface minimale** (min_area) pour éliminer les petits objets
            2. **Augmentez la constante de seuillage** (adapt_c) pour réduire la sensibilité
            3. **Activez le filtrage par circularité** et ajustez le seuil pour éliminer les formes non-insectes
            4. **Augmentez la taille du noyau morphologique** et le nombre d'itérations pour fusionner les parties fragmentées
            """)
        
        with col2:
            st.markdown("#### Insectes manquants ou incomplets")
            st.write("""
            Si certains insectes ne sont pas détectés ou sont fragmentés:
            1. **Diminuez la constante de seuillage** (adapt_c) pour augmenter la sensibilité
            2. **Diminuez la surface minimale** (min_area) pour inclure les petits insectes
            3. **Ajustez la taille du bloc adaptatif** pour qu'elle soit adaptée à la taille des insectes
            4. **Réduisez le noyau de flou gaussien** pour préserver plus de détails fins
            """)
        
        st.subheader("Guide étape par étape")
        st.write("""
        Pour obtenir les meilleurs résultats, suivez cette procédure:
        
        1. **Commencez avec une configuration prédéfinie** adaptée à vos besoins
        
        2. **Affinez le seuillage adaptatif**:
           - Ajustez `adapt_block_size` en fonction de la taille des insectes (plus grand que le plus grand insecte)
           - Modifiez `adapt_c` pour contrôler la sensibilité (augmenter si trop d'objets, diminuer si insectes manquants)
        
        3. **Optimisez les opérations morphologiques**:
           - Augmentez `morph_kernel` et `morph_iterations` si les insectes sont fragmentés
           - Réduisez ces valeurs si les insectes fusionnent entre eux ou avec d'autres objets
        
        4. **Ajustez le filtrage**:
           - Trouvez la valeur `min_area` optimale pour filtrer le bruit tout en conservant les insectes
           - Activez le filtre de circularité pour les cas difficiles
        
        5. **Vérifiez visuellement** les résultats et affinez jusqu'à obtenir une détection satisfaisante
        """)
        
        st.markdown("### Exemples de paramètres efficaces")
        
        param_examples = [
            {
                "type": "Insectes bien contrastés sur fond uniforme",
                "blur": "3-5",
                "block": "15-25",
                "c": "5-10",
                "morph": "3, 1-2 itérations",
                "area": "100-200"
            },
            {
                "type": "Insectes à faible contraste",
                "blur": "3",
                "block": "15-21",
                "c": "2-5",
                "morph": "3, 1 itération",
                "area": "50-100"
            },
            {
                "type": "Grands insectes avec détails",
                "blur": "5-7",
                "block": "31-41",
                "c": "8-12",
                "morph": "5, 2-3 itérations",
                "area": "300-500"
            }
        ]
        
        # Créer un tableau avec les exemples de paramètres
        table_data = {
            "Type d'image": [ex["type"] for ex in param_examples],
            "Flou": [ex["blur"] for ex in param_examples],
            "Bloc adaptatif": [ex["block"] for ex in param_examples],
            "Constante C": [ex["c"] for ex in param_examples],
            "Morphologie": [ex["morph"] for ex in param_examples],
            "Surface min.": [ex["area"] for ex in param_examples],
        }
        
        st.table(table_data)
        
        st.info("Astuce: Pour les cas difficiles, essayez de prétraiter vos images pour améliorer le contraste entre les insectes et le fond avant de les charger dans l'application.")

if __name__ == "__main__":
    main()
