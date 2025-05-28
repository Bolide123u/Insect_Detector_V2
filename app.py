import streamlit as st
import cv2
import numpy as np
from skimage import measure
from skimage.segmentation import clear_border
import os
import io
from PIL import Image, ImageOps
import tempfile
import zipfile
import base64
import tensorflow as tf
import matplotlib.pyplot as plt
import math

# --- Configuration Globale ---
SAVED_MODEL_DIR_PATH = "model.savedmodel"
LABELS_PATH = "labels.txt"
MODEL_INPUT_SIZE = (224, 224)

ECOLOGICAL_FUNCTIONS_MAP = {
    "Apidae": "Pollinisateurs", "Isopodes": "Décomposeurs", "Carabide": "Prédateurs",
    "Arachnides": "Prédateurs", "Mouches des semis": "Ravageur"
}
DEFAULT_ECOLOGICAL_FUNCTION = "Non défini"

DEFAULT_SEG_PARAMS = {
    "target_insect_count": 1, # Pour info, pas d'auto-tune
    "blur_kernel": 5, "adapt_block_size": 35, "adapt_c": 5, "min_area": 150,
    "morph_kernel": 3, "morph_iterations": 2, "margin": 15, "use_circularity": False,
    "min_circularity": 0.3, "apply_relative_filter": True
}

# --- Fonctions Utilitaires ---
# ... (make_square, calculate_shannon_index - inchangées)
def make_square(image, fill_color=(255, 255, 255)):
    height, width = image.shape[:2]
    max_side = max(height, width)
    top = (max_side - height) // 2
    bottom = max_side - height - top
    left = (max_side - width) // 2
    right = max_side - width - left
    square_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_color)
    return square_image

def calculate_shannon_index(counts_dict):
    if not counts_dict or sum(counts_dict.values()) == 0:
        return 0.0
    total_individuals = sum(counts_dict.values())
    shannon_index = 0.0
    for category_count in counts_dict.values():
        if category_count > 0:
            proportion = category_count / total_individuals
            shannon_index -= proportion * math.log(proportion)
    return shannon_index

# --- Fonctions de Traitement d'Image et Modèle ---
# ... (process_image, extract_insects, load_saved_model_and_labels, predict_insect_saved_model, create_label_display_image - inchangées par rapport à la version simplifiée précédente)
# Je vais les remettre ici pour que le code soit complet.
def process_image(image_cv, params):
    blur_kernel = params["blur_kernel"]
    adapt_block_size = params["adapt_block_size"]
    adapt_c = params["adapt_c"]
    min_area_param = params["min_area"]
    morph_kernel_size = params["morph_kernel"]
    morph_iterations = params["morph_iterations"]
    use_circularity = params.get("use_circularity", False)
    min_circularity = params.get("min_circularity", 0.3)
    apply_relative_filter = params.get("apply_relative_filter", True)

    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    if blur_kernel > 0:
        blur_k_odd = blur_kernel if blur_kernel % 2 != 0 else blur_kernel + 1
        blurred_img = cv2.GaussianBlur(gray, (blur_k_odd, blur_k_odd), 0)
    else:
        blurred_img = gray.copy()

    adapt_b_s_odd = adapt_block_size if adapt_block_size % 2 != 0 else adapt_block_size + 1
    if adapt_b_s_odd <= 1: adapt_b_s_odd = 3
    morph_k_odd = morph_kernel_size if morph_kernel_size % 2 != 0 else morph_kernel_size + 1
    if morph_k_odd < 1: morph_k_odd = 1
    
    thresh = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adapt_b_s_odd, adapt_c)
    kernel_closing = np.ones((morph_k_odd, morph_k_odd), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_closing, iterations=morph_iterations)
    kernel_opening = np.ones((morph_k_odd, morph_k_odd), np.uint8)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_opening, iterations=max(1, morph_iterations // 2))
    cleared = clear_border(opening)
    labels = measure.label(cleared)
    props = measure.regionprops(labels)
    
    pre_filter_props = [p for p in props if p.area >= min_area_param]

    if use_circularity:
        final_filtered_props_circ = []
        for prop_item in pre_filter_props:
            perimeter = prop_item.perimeter
            if perimeter > 0:
                circularity_val = 4 * np.pi * prop_item.area / (perimeter * perimeter)
                if circularity_val >= min_circularity:
                    final_filtered_props_circ.append(prop_item)
        filtered_props = final_filtered_props_circ
    else:
        filtered_props = pre_filter_props

    if apply_relative_filter and len(filtered_props) > 1:
        areas = [p.area for p in filtered_props]
        if areas: 
            avg_area = np.mean(areas)
            if avg_area > max(1.5 * min_area_param, 50):
                relative_threshold_area = 0.1 * avg_area
                final_relative_threshold = max(relative_threshold_area, min_area_param)
                filtered_props_after_relative = [p for p in filtered_props if p.area >= final_relative_threshold]
                filtered_props = filtered_props_after_relative
    
    return {
        "blurred": blurred_img, "thresh": thresh, "opening": opening, 
        "labels": labels, "filtered_props": filtered_props, 
        "params_used": params.copy() 
    }

def extract_insects(image, filtered_props, margin_val):
    extracted_insects = []
    for i, prop in enumerate(filtered_props):
        minr, minc, maxr, maxc = prop.bbox
        minr_marged = max(0, minr - margin_val)
        minc_marged = max(0, minc - margin_val)
        maxr_marged = min(image.shape[0], maxr + margin_val)
        maxc_marged = min(image.shape[1], maxc + margin_val)
        
        insect_roi = image[minr_marged:maxr_marged, minc_marged:maxc_marged].copy()
        roi_height, roi_width = insect_roi.shape[:2]

        if roi_height == 0 or roi_width == 0: continue
            
        mask_from_coords = np.zeros((roi_height, roi_width), dtype=np.uint8)
        for r_orig, c_orig in prop.coords:
            r_roi = r_orig - minr_marged
            c_roi = c_orig - minc_marged
            if 0 <= r_roi < roi_height and 0 <= c_roi < roi_width:
                mask_from_coords[r_roi, c_roi] = 255
        
        kernel_close_initial = np.ones((5,5), np.uint8) 
        mask_refined = cv2.morphologyEx(mask_from_coords, cv2.MORPH_CLOSE, kernel_close_initial, iterations=2) 

        contours_refined, _ = cv2.findContours(mask_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask_for_extraction = np.zeros_like(mask_refined)
        if contours_refined:
            largest_contour_refined = max(contours_refined, key=cv2.contourArea)
            cv2.drawContours(final_mask_for_extraction, [largest_contour_refined], -1, 255, thickness=cv2.FILLED)
        else: 
            final_mask_for_extraction = mask_refined 

        if np.sum(final_mask_for_extraction) == 0: continue

        mask_3ch = cv2.cvtColor(final_mask_for_extraction, cv2.COLOR_GRAY2BGR)
        white_bg = np.ones_like(insect_roi, dtype=np.uint8) * 255
        insect_on_white = np.where(mask_3ch == 255, insect_roi, white_bg)
        
        square_insect = make_square(insect_on_white, fill_color=(255, 255, 255))
        extracted_insects.append({"image": square_insect, "index": i, "original_prop": prop})
    return extracted_insects

@st.cache_resource
def load_saved_model_and_labels(model_dir_path, labels_path_arg):
    model_layer = None
    class_names_loaded = None
    try:
        abs_model_path = os.path.abspath(model_dir_path)
        if not (os.path.exists(abs_model_path) and os.path.isdir(abs_model_path) and os.path.exists(os.path.join(abs_model_path, "saved_model.pb"))):
            print(f"DEBUG: Chemin du modèle invalide ou incomplet: {abs_model_path}")
            return None, None
        model_layer = tf.keras.layers.TFSMLayer(abs_model_path, call_endpoint='serving_default')
        abs_labels_path = os.path.abspath(labels_path_arg)
        if not os.path.exists(abs_labels_path):
            print(f"DEBUG: Fichier de labels introuvable: {abs_labels_path}")
            return model_layer, None
        with open(abs_labels_path, "r") as f:
            class_names_loaded = [line.strip().split(" ", 1)[1] if " " in line.strip() else line.strip() for line in f.readlines()]
        return model_layer, class_names_loaded
    except Exception as e:
        print(f"DEBUG: Erreur chargement modèle/labels: {e}")
        return model_layer, class_names_loaded

def predict_insect_saved_model(image_cv2, model_layer_arg, class_names_arg, input_size):
    if model_layer_arg is None or class_names_arg is None:
        return "Erreur Modèle/Labels", 0.0, []
    img_resized = cv2.resize(image_cv2, input_size, interpolation=cv2.INTER_AREA)
    image_array = np.asarray(img_resized, dtype=np.float32)
    normalized_image_array = (image_array / 127.5) - 1.0
    input_tensor = tf.convert_to_tensor(normalized_image_array)
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    predictions_np = None
    try:
        predictions_output = model_layer_arg(input_tensor)
        if isinstance(predictions_output, dict):
            if len(predictions_output) == 1: predictions_tensor = list(predictions_output.values())[0]
            elif 'outputs' in predictions_output: predictions_tensor = predictions_output['outputs']
            elif 'output_0' in predictions_output: predictions_tensor = predictions_output['output_0']
            else:
                key_found = None
                for key, value in predictions_output.items():
                    if isinstance(value, tf.Tensor) and len(value.shape) == 2 and value.shape[0] == 1:
                        predictions_tensor = value; key_found = key; break
                if key_found is None: return "Erreur Sortie Modèle Dict", 0.0, []
        else: predictions_tensor = predictions_output
        if hasattr(predictions_tensor, 'numpy'): predictions_np = predictions_tensor.numpy()
        else: predictions_np = np.array(predictions_tensor)
    except Exception as e_predict: print(f"DEBUG: Erreur prédiction: {e_predict}"); return "Erreur Prédiction", 0.0, []
    if predictions_np is None or predictions_np.size == 0: return "Erreur Prédiction Vide", 0.0, []
    predicted_class_index = np.argmax(predictions_np[0])
    confidence_score = predictions_np[0][predicted_class_index]
    if predicted_class_index >= len(class_names_arg): return "Erreur Index Label", confidence_score, predictions_np[0]
    label_name = class_names_arg[predicted_class_index]
    return label_name, confidence_score, predictions_np[0]


def create_label_display_image(label_image_data, filtered_props):
    if label_image_data.ndim == 3 and label_image_data.shape[2] == 1:
        label_image_data = label_image_data.squeeze(axis=2)
    elif label_image_data.ndim != 2:
        h, w = (200, 200) if not filtered_props or not hasattr(filtered_props[0], 'image') else filtered_props[0].image.shape[:2]
        return np.zeros((h, w, 3), dtype=np.uint8)

    label_display = np.zeros((label_image_data.shape[0], label_image_data.shape[1], 3), dtype=np.uint8)
    for prop_item in filtered_props:
        color = np.random.randint(50, 256, size=3)
        for coord in prop_item.coords:
            if 0 <= coord[0] < label_display.shape[0] and 0 <= coord[1] < label_display.shape[1]:
                label_display[coord[0], coord[1]] = color
    return label_display

def main():
    st.set_page_config(layout="wide")
    st.title("Détection, isolation et identification dʼinsectes")

    # Initialisation de st.session_state
    if 'image_data_list' not in st.session_state: st.session_state.image_data_list = []
    if 'model_obj' not in st.session_state: st.session_state.model_obj = None
    if 'class_names_list' not in st.session_state: st.session_state.class_names_list = None
    if 'active_image_id_for_params' not in st.session_state: st.session_state.active_image_id_for_params = None
    if 'first_model_load_message' not in st.session_state: st.session_state.first_model_load_message = False


    if st.session_state.model_obj is None:
        model_loaded, class_names_loaded = load_saved_model_and_labels(SAVED_MODEL_DIR_PATH, LABELS_PATH)
        if model_loaded:
            st.session_state.model_obj = model_loaded
            if class_names_loaded:
                st.session_state.class_names_list = class_names_loaded
                if not st.session_state.first_model_load_message:
                    st.success("Modèle d'identification et labels chargés avec succès !")
                    st.session_state.first_model_load_message = True
            else: st.warning("Modèle chargé, mais échec du chargement des labels.")
        # else: st.error("Échec du chargement du modèle d'identification.") # Affiché par load_saved_model_and_labels via print
    
    model_to_use = st.session_state.model_obj
    class_names_to_use = st.session_state.class_names_list

    # --- Sidebar pour les paramètres de l'image active ---
    with st.sidebar:
        st.header("Paramètres de Segmentation")
        active_id = st.session_state.active_image_id_for_params
        
        active_img_data = None
        if active_id:
            try:
                active_img_data = next(item for item in st.session_state.image_data_list if item["id"] == active_id)
            except StopIteration: # Si l'ID n'est plus valide (ex: image supprimée)
                st.session_state.active_image_id_for_params = None
                active_id = None # Réinitialiser active_id aussi

        if active_img_data:
            st.markdown(f"**Pour : {active_img_data['filename']}**")
            params_sidebar = active_img_data["params"]

            # Pas de target_insect_count pour l'auto-tune
            # params_sidebar["target_insect_count"] = st.number_input("Insectes attendus (info)", 0,100, params_sidebar["target_insect_count"],1, key=f"sb_target_{active_id}")
            params_sidebar["blur_kernel"] = st.slider("Flou (0=aucun)", 0, 21, params_sidebar["blur_kernel"], 1, key=f"sb_blur_{active_id}")
            params_sidebar["adapt_block_size"] = st.slider("Bloc Adapt.", 3, 51, params_sidebar["adapt_block_size"], 2, key=f"sb_block_{active_id}")
            params_sidebar["adapt_c"] = st.slider("Constante C", -20, 20, params_sidebar["adapt_c"], 1, key=f"sb_c_{active_id}")
            params_sidebar["min_area"] = st.slider("Aire Min", 10, 10000, params_sidebar["min_area"], 10, key=f"sb_area_{active_id}")
            params_sidebar["morph_kernel"] = st.slider("Noyau Morpho", 1, 15, params_sidebar["morph_kernel"], 2, key=f"sb_morph_k_{active_id}")
            params_sidebar["morph_iterations"] = st.slider("It. Morpho", 1, 5, params_sidebar["morph_iterations"], 1, key=f"sb_morph_i_{active_id}")
            params_sidebar["margin"] = st.slider("Marge Ext.", 0, 50, params_sidebar["margin"], key=f"sb_margin_{active_id}")
            params_sidebar["use_circularity"] = st.checkbox("Filtre Circ.", params_sidebar["use_circularity"], key=f"sb_circ_c_{active_id}")
            if params_sidebar["use_circularity"]:
                params_sidebar["min_circularity"] = st.slider("Circ. Min Val", 0.0, 1.0, params_sidebar["min_circularity"], 0.05, key=f"sb_circ_v_{active_id}")
            params_sidebar["apply_relative_filter"] = st.checkbox("Filtre Relatif", params_sidebar["apply_relative_filter"], key=f"sb_rel_f_{active_id}")

            if st.button("Appliquer et Traiter l'Image Active", key=f"sb_apply_btn_{active_id}"):
                with st.spinner(f"Traitement de {active_img_data['filename']}..."):
                    active_img_data["processed_data"] = process_image(
                        active_img_data["cv_image"], 
                        params_sidebar # Utilise les params modifiés de la sidebar
                    )
                    active_img_data["is_processed"] = True
                    # Pas besoin de rerun ici si on veut que la sidebar reste et que l'onglet principal se mette à jour
                    # Cependant, pour que l'affichage DANS L'ONGLET se mette à jour, un rerun est nécessaire
                    st.rerun() 
        else:
            st.info("Sélectionnez une image dans l'onglet 'Segmentation' (bouton 'Configurer') pour ajuster ses paramètres ici.")


    tab1, tab2, tab3 = st.tabs(["Segmentation par Image", "Analyse Globale", "Guide"])

    with tab1:
        st.header("Configuration et Segmentation Image par Image")
        
        uploaded_files = st.file_uploader(
            "1. Choisissez vos images", type=["jpg", "jpeg", "png"],
            accept_multiple_files=True, key="tab1_main_file_uploader_key"
        )

        if uploaded_files:
            new_uploaded_file_ids = {f.file_id + "_" + f.name for f in uploaded_files}
            existing_img_data_map = {img_d["id"]: img_d for img_d in st.session_state.image_data_list}
            
            updated_image_data_list_tab1 = []
            files_changed = False

            for uploaded_file_item_tab1 in uploaded_files:
                img_id_new_tab1 = uploaded_file_item_tab1.file_id + "_" + uploaded_file_item_tab1.name
                if img_id_new_tab1 in existing_img_data_map:
                    updated_image_data_list_tab1.append(existing_img_data_map[img_id_new_tab1])
                else: # Nouvelle image
                    files_changed = True
                    img_bytes_new_tab1 = uploaded_file_item_tab1.getvalue()
                    img_cv_new_tab1 = cv2.imdecode(np.frombuffer(img_bytes_new_tab1, np.uint8), cv2.IMREAD_COLOR)
                    updated_image_data_list_tab1.append({
                        "id": img_id_new_tab1, "filename": uploaded_file_item_tab1.name,
                        "image_bytes": img_bytes_new_tab1, "cv_image": img_cv_new_tab1,
                        "params": DEFAULT_SEG_PARAMS.copy(),
                        "processed_data": None, "is_processed": False
                    })
            
            if len(updated_image_data_list_tab1) != len(st.session_state.image_data_list) or files_changed:
                files_changed = True # Marquer si des images ont été supprimées

            st.session_state.image_data_list = updated_image_data_list_tab1
            
            if files_changed and st.session_state.image_data_list: # Si les fichiers ont changé et qu'il y en a
                st.session_state.active_image_id_for_params = st.session_state.image_data_list[0]["id"] # Activer la première
                st.rerun()
            elif not st.session_state.image_data_list: # Si tous les fichiers ont été retirés
                 st.session_state.active_image_id_for_params = None
                 st.rerun()


        if not st.session_state.image_data_list:
            st.info("Veuillez téléverser des images.")
        
        if st.session_state.image_data_list:
            st.markdown("---")
            if st.button("Traiter TOUTES les images (avec leurs paramètres respectifs)", key="process_all_btn_tab1"):
                num_images_proc_all = len(st.session_state.image_data_list)
                prog_bar_all = st.progress(0)
                stat_text_all = st.empty()
                for i_all, img_data_proc_all in enumerate(st.session_state.image_data_list):
                    stat_text_all.text(f"Traitement de {img_data_proc_all['filename']} ({i_all+1}/{num_images_proc_all})...")
                    img_data_proc_all["processed_data"] = process_image(
                        img_data_proc_all["cv_image"], 
                        img_data_proc_all["params"]
                    )
                    img_data_proc_all["is_processed"] = True
                    prog_bar_all.progress((i_all + 1) / num_images_proc_all)
                stat_text_all.success("Toutes les images ont été traitées.")
                # st.rerun() # Le rerun est implicite car l'affichage va se mettre à jour

        for idx_tab1, img_data_item_tab1 in enumerate(st.session_state.image_data_list):
            st.markdown(f"--- \n ### Image {idx_tab1 + 1}: {img_data_item_tab1['filename']}")
            
            # Bouton "Configurer" plus visible (toggle)
            # On va utiliser un bouton simple pour sélectionner l'image active pour la sidebar
            select_button_key = f"select_for_sidebar_btn_{img_data_item_tab1['id']}"
            if st.button(f"⚙️ Configurer {img_data_item_tab1['filename']}", key=select_button_key):
                st.session_state.active_image_id_for_params = img_data_item_tab1["id"]
                st.info(f"'{img_data_item_tab1['filename']}' sélectionnée. Modifiez ses paramètres dans la barre latérale gauche et cliquez sur 'Appliquer et Traiter'.")
                # Pas de rerun ici, la sidebar devrait se mettre à jour lors de sa prochaine exécution
                # ou on peut forcer un rerun si la sidebar ne se met pas à jour immédiatement.
                # st.rerun()


            cols_img_display_item_tab1 = st.columns(3)
            with cols_img_display_item_tab1[0]:
                st.image(cv2.cvtColor(img_data_item_tab1["cv_image"], cv2.COLOR_BGR2RGB), caption="Originale", use_column_width=True)
            
            if img_data_item_tab1["is_processed"] and img_data_item_tab1["processed_data"]:
                with cols_img_display_item_tab1[1]:
                    st.image(img_data_item_tab1["processed_data"]["opening"], channels="GRAY", caption="Résultat Morphologique", use_column_width=True)
                with cols_img_display_item_tab1[2]:
                    label_disp_img_main_item_tab1 = create_label_display_image(img_data_item_tab1["processed_data"]["labels"], img_data_item_tab1["processed_data"]["filtered_props"])
                    num_detected_item_tab1 = len(img_data_item_tab1['processed_data']['filtered_props'])
                    st.image(label_disp_img_main_item_tab1, caption=f"Détectés: {num_detected_item_tab1}", use_column_width=True)
                st.metric(label=f"Insectes Détectés", value=num_detected_item_tab1)
            else:
                with cols_img_display_item_tab1[1]:
                    st.caption("Résultat morphologique apparaîtra ici après traitement.")
                with cols_img_display_item_tab1[2]:
                    st.caption("Labels colorés apparaîtront ici après traitement.")


    with tab2:
        st.header("Analyse Globale des Insectes Identifiés")
        # ... (Logique de l'onglet 2, s'assurer que la taille du pie chart est bien petite)
        if model_to_use is None or class_names_to_use is None:
            st.error("Modèle d'identification ou labels non disponibles.")
        elif not st.session_state.image_data_list or not any(img_d_tab2["is_processed"] for img_d_tab2 in st.session_state.image_data_list):
            st.info("Veuillez d'abord téléverser et traiter des images (Onglet 'Segmentation par Image').")
        else:
            all_identified_labels_for_pie_chart_tab2 = []
            images_processed_for_id_tab2 = [img_tab2 for img_tab2 in st.session_state.image_data_list if img_tab2["is_processed"] and img_tab2["processed_data"]]

            if not images_processed_for_id_tab2:
                 st.info("Aucune image n'a été traitée avec succès pour l'identification.")
            else:
                st.write(f"Analyse basée sur {len(images_processed_for_id_tab2)} image(s) traitée(s).")
                for img_data_item_id_tab2 in images_processed_for_id_tab2:
                    # S'assurer que les clés existent avant de les utiliser
                    if "cv_image" not in img_data_item_id_tab2 or \
                       "processed_data" not in img_data_item_id_tab2 or \
                       "filtered_props" not in img_data_item_id_tab2["processed_data"] or \
                       "params" not in img_data_item_id_tab2 or \
                       "margin" not in img_data_item_id_tab2["params"]:
                        st.warning(f"Données manquantes pour l'image {img_data_item_id_tab2.get('filename', 'inconnue')}, elle sera ignorée pour l'identification.")
                        continue # Passer à l'image suivante

                    extracted_insects_id_list_tab2 = extract_insects(
                        img_data_item_id_tab2["cv_image"], 
                        img_data_item_id_tab2["processed_data"]["filtered_props"], 
                        img_data_item_id_tab2["params"]["margin"]
                    )
                    for insect_item_tab2 in extracted_insects_id_list_tab2:
                        label_id_val_tab2, _, _ = predict_insect_saved_model(
                            insect_item_tab2["image"], model_to_use, class_names_to_use, MODEL_INPUT_SIZE
                        )
                        if "Erreur" not in label_id_val_tab2:
                            all_identified_labels_for_pie_chart_tab2.append(label_id_val_tab2)
                
                if not all_identified_labels_for_pie_chart_tab2:
                    st.warning("Aucun insecte n'a pu être identifié sur l'ensemble des images.")
                else:
                    ecological_counts_for_pie_tab2 = {}
                    labels_non_mappes_pie_glob_tab2 = set()
                    for label_pie_item_tab2 in all_identified_labels_for_pie_chart_tab2:
                        if label_pie_item_tab2 not in ECOLOGICAL_FUNCTIONS_MAP:
                            labels_non_mappes_pie_glob_tab2.add(label_pie_item_tab2)
                        eco_func_item_tab2 = ECOLOGICAL_FUNCTIONS_MAP.get(label_pie_item_tab2, DEFAULT_ECOLOGICAL_FUNCTION)
                        ecological_counts_for_pie_tab2[eco_func_item_tab2] = ecological_counts_for_pie_tab2.get(eco_func_item_tab2, 0) + 1
                    
                    if labels_non_mappes_pie_glob_tab2:
                        st.warning(f"Labels globaux non mappés pour pie chart: {labels_non_mappes_pie_glob_tab2}")

                    if ecological_counts_for_pie_tab2:
                        st.subheader("Répartition Globale des Fonctions Écologiques")
                        labels_pie_chart_keys_tab2 = list(ecological_counts_for_pie_tab2.keys())
                        sizes_pie_chart_values_tab2 = list(ecological_counts_for_pie_tab2.values())
                        colors_map_for_pie_tab2 = {"Décomposeurs": "#8B4513", "Pollinisateurs": "#FFD700", "Prédateurs": "#DC143C", "Ravageur": "#FF8C00", "Non défini": "#D3D3D3"}
                        pie_colors_for_chart_tab2 = [colors_map_for_pie_tab2.get(lbl_p_tab2, "#CCCCCC") for lbl_p_tab2 in labels_pie_chart_keys_tab2]
                        
                        # MODIFICATION : Taille du Pie Chart encore plus réduite
                        fig_pie_chart_tab2, ax_pie_chart_tab2 = plt.subplots(figsize=(4, 2.8)) # ex: 4x2.8 ou 3.5x2.5
                        ax_pie_chart_tab2.pie(sizes_pie_chart_values_tab2, labels=labels_pie_chart_keys_tab2, autopct='%1.1f%%', startangle=90, colors=pie_colors_for_chart_tab2, textprops={'fontsize': 6}) # Police encore plus petite
                        ax_pie_chart_tab2.axis('equal')
                        st.pyplot(fig_pie_chart_tab2)

                        shannon_idx_val_tab2 = calculate_shannon_index(ecological_counts_for_pie_tab2)
                        st.subheader("Indice de Shannon Fonctionnel Global (H')")
                        st.metric(label="H'", value=f"{shannon_idx_val_tab2:.3f}")
                        # ... (captions pour Shannon)
                        if shannon_idx_val_tab2 == 0 and sum(ecological_counts_for_pie_tab2.values()) > 0:
                            st.caption("Un indice de 0 signifie qu'une seule fonction écologique est présente.")
                        elif shannon_idx_val_tab2 > 0:
                            max_shannon_val_info_tab2 = math.log(len(ecological_counts_for_pie_tab2)) if len(ecological_counts_for_pie_tab2) > 0 else 0
                            st.caption(f"Max H' possible pour {len(ecological_counts_for_pie_tab2)} fonctions: {max_shannon_val_info_tab2:.3f}.")

                    else:
                        st.write("Aucune fonction écologique à afficher.")
            
            st.markdown("--- \n ### Identification Détaillée par Image")
            # ... (Affichage détaillé par image comme avant)
            for idx_detail_tab2, img_data_item_detail_id_tab2 in enumerate(images_processed_for_id_tab2): # Utiliser la liste filtrée
                st.markdown(f"#### {img_data_item_detail_id_tab2['filename']}")
                # S'assurer que les clés existent avant de les utiliser
                if "cv_image" not in img_data_item_detail_id_tab2 or \
                   "processed_data" not in img_data_item_detail_id_tab2 or \
                   "filtered_props" not in img_data_item_detail_id_tab2["processed_data"] or \
                   "params" not in img_data_item_detail_id_tab2 or \
                   "margin" not in img_data_item_detail_id_tab2["params"]:
                    st.warning(f"Données de segmentation manquantes pour {img_data_item_detail_id_tab2.get('filename', 'inconnue')}, identification détaillée ignorée.")
                    continue

                extracted_insects_detail_id_tab2 = extract_insects(
                    img_data_item_detail_id_tab2["cv_image"], 
                    img_data_item_detail_id_tab2["processed_data"]["filtered_props"], 
                    img_data_item_detail_id_tab2["params"]["margin"]
                )
                if not extracted_insects_detail_id_tab2:
                    st.write("Aucun insecte extrait pour identification sur cette image.")
                    continue
                
                num_cols_id_detail_disp_tab2 = 3
                cols_id_detail_disp_tab2 = st.columns(num_cols_id_detail_disp_tab2)
                col_idx_id_detail_disp_tab2 = 0
                for insect_detail_item_id_tab2 in extracted_insects_detail_id_tab2:
                    label_detail_id_tab2, confidence_detail_id_tab2, _ = predict_insect_saved_model(
                        insect_detail_item_id_tab2["image"], model_to_use, class_names_to_use, MODEL_INPUT_SIZE
                    )
                    with cols_id_detail_disp_tab2[col_idx_id_detail_disp_tab2 % num_cols_id_detail_disp_tab2]:
                        st.image(cv2.cvtColor(insect_detail_item_id_tab2["image"], cv2.COLOR_BGR2RGB), caption=f"Insecte #{insect_detail_item_id_tab2['index'] + 1}", width=150)
                        if "Erreur" in label_detail_id_tab2:
                            st.error(f"{label_detail_id_tab2} ({confidence_detail_id_tab2*100:.2f}%)")
                        else:
                            st.markdown(f"**Label:** {label_detail_id_tab2}")
                            st.markdown(f"**Fonction:** {ECOLOGICAL_FUNCTIONS_MAP.get(label_detail_id_tab2, DEFAULT_ECOLOGICAL_FUNCTION)}")
                            st.markdown(f"**Confiance:** {confidence_detail_id_tab2*100:.2f}%")
                    col_idx_id_detail_disp_tab2 += 1
                st.markdown("---")

    with tab3:
        st.header("Guide dʼutilisation")
        st.subheader("Segmentation par Image (Onglet 1)")
        st.write("""
        1.  **Téléversez vos images.**
        2.  Pour chaque image, un bouton "⚙️ Configurer [nom_image]" apparaît. Cliquez dessus pour rendre cette image 'active'.
        3.  Les paramètres de l'image active s'affichent et peuvent être modifiés dans la **barre latérale de gauche**.
        4.  Dans la sidebar, cliquez sur **"Appliquer et Traiter l'Image Active"** pour traiter l'image avec les nouveaux réglages.
        5.  Un bouton "Traiter TOUTES les images" est disponible en haut de la liste des images pour lancer la segmentation sur l'ensemble du lot, chacune avec ses propres paramètres configurés.
        """)
        st.subheader("Analyse Globale (Onglet 2)")
        st.write("""
        Cet onglet s'active une fois qu'au moins une image a été traitée. Il affiche:
        - Un graphique circulaire (camembert) de la répartition globale des fonctions écologiques.
        - L'Indice de Shannon Fonctionnel global.
        - Une identification détaillée par image.
        """)


if __name__ == "__main__":
    if 'image_data_list' not in st.session_state: st.session_state.image_data_list = []
    if 'model_obj' not in st.session_state: st.session_state.model_obj = None
    if 'class_names_list' not in st.session_state: st.session_state.class_names_list = None
    if 'active_image_id_for_params' not in st.session_state: st.session_state.active_image_id_for_params = None
    if 'first_model_load_message' not in st.session_state: st.session_state.first_model_load_message = False
    main()
