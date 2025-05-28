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
import pandas as pd

# --- Configuration Globale ---
SAVED_MODEL_DIR_PATH = "model.savedmodel"
LABELS_PATH = "labels.txt"
MODEL_INPUT_SIZE = (224, 224)

ECOLOGICAL_FUNCTIONS_MAP = {
    "Apidae": "Pollinisateurs",
    "Isopodes": "Décomposeurs et Ingénieurs du sol",
    "Carabide": "Prédateurs", # Note: Model might output "Carabidae"
    "Carabidae": "Prédateurs", # Added for consistency if model outputs this
    "Opiliones et Araneae": "Prédateurs",
    "Mouches des semis": "Ravageur"
}
DEFAULT_ECOLOGICAL_FUNCTION = "Non défini"

DEFAULT_SEG_PARAMS = {
    "target_insect_count": 1,
    "blur_kernel": 5, "adapt_block_size": 35, "adapt_c": 5, "min_area": 150,
    "morph_kernel": 3, "morph_iterations": 2, "margin": 15, "use_circularity": False,
    "min_circularity": 0.3, "apply_relative_filter": True
}

# Liste des termes latins à mettre en italique (sensible à la casse)
LATIN_TERMS_FOR_ITALICS = ["Apidae", "Isopodes", "Carabide", "Opiliones", "Araneae", "Carabidae", "Andrenidae"]


# --- Fonctions Utilitaires ---
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
    if not counts_dict or sum(counts_dict.values()) == 0: return 0.0
    total_individuals = sum(counts_dict.values())
    shannon_index = 0.0
    for category_count in counts_dict.values():
        if category_count > 0:
            proportion = category_count / total_individuals
            shannon_index -= proportion * math.log(proportion)
    return shannon_index

# --- Fonctions de Traitement d'Image et Modèle ---
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

def extract_arthropods(image, filtered_props, margin_val):
    extracted_arthropods_list = []
    for i, prop in enumerate(filtered_props):
        minr, minc, maxr, maxc = prop.bbox
        minr_marged = max(0, minr - margin_val)
        minc_marged = max(0, minc - margin_val)
        maxr_marged = min(image.shape[0], maxr + margin_val)
        maxc_marged = min(image.shape[1], maxc + margin_val)
        
        arthropod_roi = image[minr_marged:maxr_marged, minc_marged:maxc_marged].copy()
        roi_height, roi_width = arthropod_roi.shape[:2]

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
            if cv2.contourArea(largest_contour_refined) > 5: 
                 cv2.drawContours(final_mask_for_extraction, [largest_contour_refined], -1, 255, thickness=cv2.FILLED)
        else: 
            if np.sum(mask_refined) > 5 * 255 : 
                final_mask_for_extraction = mask_refined 

        if np.sum(final_mask_for_extraction) == 0: continue 

        mask_3ch = cv2.cvtColor(final_mask_for_extraction, cv2.COLOR_GRAY2BGR)
        white_bg = np.ones_like(arthropod_roi, dtype=np.uint8) * 255 
        arthropod_on_white = np.where(mask_3ch == 255, arthropod_roi, white_bg) 
        
        square_arthropod = make_square(arthropod_on_white, fill_color=(255, 255, 255)) 
        extracted_arthropods_list.append({"image": square_arthropod, "index": i, "original_prop": prop}) 
    return extracted_arthropods_list

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
        with open(abs_labels_path, "r", encoding="utf-8") as f:
            class_names_raw = [line.strip() for line in f.readlines()]
            class_names_loaded = []
            for line in class_names_raw:
                parts = line.split(" ", 1)
                if len(parts) > 1 and parts[0].isdigit():
                    class_names_loaded.append(parts[1]) 
                else:
                    class_names_loaded.append(line) 
        return model_layer, class_names_loaded
    except Exception as e:
        print(f"DEBUG: Erreur chargement modèle/labels: {e}")
        return model_layer, class_names_loaded 

def predict_arthropod_saved_model(image_cv2, model_layer_arg, class_names_arg, input_size):
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
        h, w = (200,200) if not filtered_props or not hasattr(filtered_props[0],'image') else filtered_props[0].image.shape[:2]
        return np.zeros((h, w, 3), dtype=np.uint8)

    label_display = np.zeros((label_image_data.shape[0], label_image_data.shape[1], 3), dtype=np.uint8)
    for prop_item in filtered_props:
        color = np.random.randint(50, 256, size=3) 
        for coord in prop_item.coords: 
            if 0 <= coord[0] < label_display.shape[0] and 0 <= coord[1] < label_display.shape[1]:
                label_display[coord[0], coord[1]] = color
    return label_display

def italicize_latin_terms(text, latin_terms):
    for term in latin_terms:
        if term in text: 
             text = text.replace(term, f"*{term}*")
    return text

# Helper function to convert DataFrame to CSV for download
@st.cache_data
def convert_df_to_csv(df_to_convert):
    return df_to_convert.to_csv(index=False).encode('utf-8')

def main():
    st.set_page_config(layout="wide")
    st.title("Détection, isolation et identification dʼArthropodes")

    if 'image_data_list' not in st.session_state: st.session_state.image_data_list = []
    if 'model_obj' not in st.session_state: st.session_state.model_obj = None
    if 'class_names_list' not in st.session_state: st.session_state.class_names_list = None
    if 'active_image_id_for_params' not in st.session_state: st.session_state.active_image_id_for_params = None
    if 'first_model_load_message_displayed' not in st.session_state: st.session_state.first_model_load_message_displayed = False

    if st.session_state.model_obj is None:
        model_loaded, class_names_loaded = load_saved_model_and_labels(SAVED_MODEL_DIR_PATH, LABELS_PATH)
        if model_loaded:
            st.session_state.model_obj = model_loaded
            if class_names_loaded:
                st.session_state.class_names_list = class_names_loaded
                if not st.session_state.first_model_load_message_displayed: 
                    st.sidebar.success("Modèle et labels chargés !")
                    st.session_state.first_model_load_message_displayed = True
            else: st.sidebar.warning("Modèle chargé, mais échec du chargement des labels.")
    
    model_to_use = st.session_state.model_obj
    class_names_to_use = st.session_state.class_names_list

    with st.sidebar:
        st.header("Paramètres de Segmentation")
        active_id_sidebar = st.session_state.active_image_id_for_params
        active_img_data_sb = None 
        if active_id_sidebar:
            try: 
                active_img_data_sb = next(item for item in st.session_state.image_data_list if item["id"] == active_id_sidebar)
            except StopIteration: 
                st.session_state.active_image_id_for_params = None; active_id_sidebar = None 

        if active_img_data_sb: 
            st.markdown(f"**Pour : {active_img_data_sb['filename']}**")
            params_sb_ref = active_img_data_sb["params"] 
            params_sb_ref["blur_kernel"]=st.slider("Flou (0=aucun)",0,21,params_sb_ref["blur_kernel"],1,key=f"sb_blur_{active_id_sidebar}")
            params_sb_ref["adapt_block_size"]=st.slider("Bloc Adapt.",3,51,params_sb_ref["adapt_block_size"],2,key=f"sb_block_{active_id_sidebar}")
            params_sb_ref["adapt_c"]=st.slider("Constante C",-20,20,params_sb_ref["adapt_c"],1,key=f"sb_c_{active_id_sidebar}")
            params_sb_ref["min_area"]=st.slider("Aire Min",10,10000,params_sb_ref["min_area"],10,key=f"sb_area_{active_id_sidebar}")
            params_sb_ref["morph_kernel"]=st.slider("Noyau Morpho",1,15,params_sb_ref["morph_kernel"],2,key=f"sb_morph_k_{active_id_sidebar}")
            params_sb_ref["morph_iterations"]=st.slider("It. Morpho",1,5,params_sb_ref["morph_iterations"],1,key=f"sb_morph_i_{active_id_sidebar}")
            params_sb_ref["margin"]=st.slider("Marge Ext.",0,50,params_sb_ref["margin"],key=f"sb_margin_{active_id_sidebar}")
            params_sb_ref["use_circularity"]=st.checkbox("Filtre Circ.",params_sb_ref["use_circularity"],key=f"sb_circ_c_{active_id_sidebar}")
            if params_sb_ref["use_circularity"]:
                params_sb_ref["min_circularity"]=st.slider("Circ. Min Val",0.0,1.0,params_sb_ref["min_circularity"],0.05,key=f"sb_circ_v_{active_id_sidebar}")
            params_sb_ref["apply_relative_filter"]=st.checkbox("Filtre Relatif",params_sb_ref["apply_relative_filter"],key=f"sb_rel_f_{active_id_sidebar}")

            if st.button("Appliquer & Traiter Image Active", key=f"sb_apply_btn_v4_{active_id_sidebar}"):
                with st.spinner(f"Traitement de {active_img_data_sb['filename']}..."):
                    active_img_data_sb["processed_data"] = process_image(active_img_data_sb["cv_image"], params_sb_ref)
                    active_img_data_sb["is_processed"] = True
                st.rerun() 
        else:
            st.info("Sélectionnez une image (bouton '⚙️ Configurer') pour ajuster ses paramètres ici.")

    tab1, tab2, tab3 = st.tabs(["Segmentation par Image", "Analyse Globale", "Guide"])

    with tab1:
        st.header("Configuration et Segmentation Image par Image")
        uploaded_files_tab1_main_v5 = st.file_uploader(
            "1. Choisissez vos images", type=["jpg", "jpeg", "png"],
            accept_multiple_files=True, key="tab1_file_uploader_main_v5"
        )
        
        if uploaded_files_tab1_main_v5:
            new_ids_uploaded_tab1_v5 = {f.file_id + "_" + f.name for f in uploaded_files_tab1_main_v5} 
            existing_map_in_session_tab1_v5 = {img_d_sess["id"]: img_d_sess for img_d_sess in st.session_state.image_data_list}
            updated_list_session_tab1_v5 = [] 
            files_changed_in_session_tab1_v5 = False

            for up_file_item_tab1_v5 in uploaded_files_tab1_main_v5:
                img_id_up_item_tab1_v5 = up_file_item_tab1_v5.file_id + "_" + up_file_item_tab1_v5.name
                if img_id_up_item_tab1_v5 in existing_map_in_session_tab1_v5: 
                    updated_list_session_tab1_v5.append(existing_map_in_session_tab1_v5[img_id_up_item_tab1_v5])
                else: 
                    files_changed_in_session_tab1_v5 = True
                    bytes_up_item_tab1_v5 = up_file_item_tab1_v5.getvalue()
                    cv_img_up_item_tab1_v5 = cv2.imdecode(np.frombuffer(bytes_up_item_tab1_v5, np.uint8), cv2.IMREAD_COLOR)
                    updated_list_session_tab1_v5.append({
                        "id": img_id_up_item_tab1_v5, "filename": up_file_item_tab1_v5.name,
                        "image_bytes": bytes_up_item_tab1_v5, "cv_image": cv_img_up_item_tab1_v5,
                        "params": DEFAULT_SEG_PARAMS.copy(), 
                        "processed_data": None, "is_processed": False
                    })
            if len(updated_list_session_tab1_v5) != len(st.session_state.image_data_list): files_changed_in_session_tab1_v5 = True
            st.session_state.image_data_list = updated_list_session_tab1_v5
            if files_changed_in_session_tab1_v5: 
                if st.session_state.active_image_id_for_params not in new_ids_uploaded_tab1_v5:
                     st.session_state.active_image_id_for_params = st.session_state.image_data_list[0]["id"] if st.session_state.image_data_list else None
                st.rerun()

        if not st.session_state.image_data_list: st.info("Veuillez téléverser des images.")
        
        if st.session_state.image_data_list:
            st.markdown("---")
            button_style = """
                <style>
                .big-button {
                    display: inline-block;
                    padding: 0.75em 1.5em;
                    font-size: 1.2em;
                    font-weight: bold;
                    color: white;
                    background-color: #FF4B4B; 
                    border: none;
                    border-radius: 0.5em;
                    text-align: center;
                    text-decoration: none;
                    cursor: pointer;
                }
                .big-button:hover {
                    background-color: #FF6B6B;
                }
                </style>
            """
            st.markdown(button_style, unsafe_allow_html=True) 
            
            col_btn1, col_btn2, col_btn3 = st.columns([1,2,1]) 
            with col_btn2:
                if st.button("▶️ Segmenter les Images (avec leurs paramètres individuels)", key="process_all_btn_tab1_v5_styled", use_container_width=True):
                    num_all_proc_tab1_v5 = len(st.session_state.image_data_list)
                    if num_all_proc_tab1_v5 > 0:
                        prog_bar_all_tab1_v5 = st.progress(0)
                        stat_all_tab1_v5 = st.empty() 
                        for i_all_p_tab1_v5, img_d_p_all_tab1_v5 in enumerate(st.session_state.image_data_list):
                            stat_all_tab1_v5.text(f"Traitement de {img_d_p_all_tab1_v5['filename']} ({i_all_p_tab1_v5+1}/{num_all_proc_tab1_v5})...")
                            img_d_p_all_tab1_v5["processed_data"] = process_image(img_d_p_all_tab1_v5["cv_image"], img_d_p_all_tab1_v5["params"])
                            img_d_p_all_tab1_v5["is_processed"] = True
                            prog_bar_all_tab1_v5.progress((i_all_p_tab1_v5+1)/num_all_proc_tab1_v5)
                        stat_all_tab1_v5.success("Toutes les images ont été traitées.")
                    else:
                        st.warning("Aucune image à traiter.")


        for idx_main_tab1_disp_v5, img_data_main_tab1_disp_v5 in enumerate(st.session_state.image_data_list):
            st.markdown(f"--- \n ### Image {idx_main_tab1_disp_v5 + 1}: {img_data_main_tab1_disp_v5['filename']}")
            select_btn_key_main_tab1_disp_v5 = f"select_cfg_btn_main_tab1_disp_v5_{img_data_main_tab1_disp_v5['id']}"
            
            col_img_main_tab1_disp_v5, col_actions_main_tab1_disp_v5 = st.columns([4,1.2]) 
            with col_img_main_tab1_disp_v5:
                 st.image(cv2.cvtColor(img_data_main_tab1_disp_v5["cv_image"], cv2.COLOR_BGR2RGB), caption="Originale", use_column_width=True)
            with col_actions_main_tab1_disp_v5:
                st.markdown("<br>", unsafe_allow_html=True) 
                if st.button(f"⚙️ Configurer", key=select_btn_key_main_tab1_disp_v5, help=f"Éditer les paramètres pour {img_data_main_tab1_disp_v5['filename']}", use_container_width=True):
                    st.session_state.active_image_id_for_params = img_data_main_tab1_disp_v5["id"]
                    st.rerun() 

            if img_data_main_tab1_disp_v5["is_processed"] and img_data_main_tab1_disp_v5["processed_data"]:
                cols_res_main_tab1_disp_v5 = st.columns(2)
                with cols_res_main_tab1_disp_v5[0]:
                    st.image(img_data_main_tab1_disp_v5["processed_data"]["opening"], channels="GRAY", caption="Résultat Morphologique", use_column_width=True)
                with cols_res_main_tab1_disp_v5[1]:
                    label_disp_main_tab1_val_v5 = create_label_display_image(img_data_main_tab1_disp_v5["processed_data"]["labels"], img_data_main_tab1_disp_v5["processed_data"]["filtered_props"])
                    num_det_main_tab1_val_v5 = len(img_data_main_tab1_disp_v5['processed_data']['filtered_props'])
                    st.image(label_disp_main_tab1_val_v5, caption=f"Arthropodes Détectés: {num_det_main_tab1_val_v5}", use_column_width=True)
                st.metric(label=f"Arthropodes Détectés", value=num_det_main_tab1_val_v5) 
            else: st.caption("Résultats de segmentation apparaîtront ici après traitement.")

    with tab2:
        st.header("Analyse Globale des Arthropodes Identifiés")
        if model_to_use is None or class_names_to_use is None:
            st.error("Modèle d'identification ou labels non disponibles. Vérifiez le chargement au démarrage.")
        elif not st.session_state.image_data_list or not any(img_d_tab2_an["is_processed"] for img_d_tab2_an in st.session_state.image_data_list):
            st.info("Veuillez d'abord traiter des images (Onglet 'Segmentation par Image').")
        else:
            all_labels_pie_tab2_an_list_v5 = []
            imgs_processed_tab2_an_list_v5 = [img_tab2_an for img_tab2_an in st.session_state.image_data_list if img_tab2_an["is_processed"] and img_tab2_an["processed_data"]]

            if not imgs_processed_tab2_an_list_v5: st.info("Aucune image traitée disponible pour l'identification.")
            else:
                st.write(f"Analyse basée sur {len(imgs_processed_tab2_an_list_v5)} image(s) traitée(s).")
                for img_item_tab2_an_val_v5 in imgs_processed_tab2_an_list_v5:
                    if not (img_item_tab2_an_val_v5.get("cv_image") is not None and 
                            img_item_tab2_an_val_v5.get("processed_data", {}).get("filtered_props") is not None and
                            img_item_tab2_an_val_v5.get("params", {}).get("margin") is not None):
                        continue 
                    extracted_arthropods_tab2_an_list_v5 = extract_arthropods(
                        img_item_tab2_an_val_v5["cv_image"], img_item_tab2_an_val_v5["processed_data"]["filtered_props"], 
                        img_item_tab2_an_val_v5["params"]["margin"]
                    )
                    for arthropod_tab2_an_item_v5 in extracted_arthropods_tab2_an_list_v5:
                        label_val_tab2_an_item_v5, _, _ = predict_arthropod_saved_model(
                            arthropod_tab2_an_item_v5["image"], model_to_use, class_names_to_use, MODEL_INPUT_SIZE
                        )
                        if "Erreur" not in label_val_tab2_an_item_v5: all_labels_pie_tab2_an_list_v5.append(label_val_tab2_an_item_v5)
                
                if not all_labels_pie_tab2_an_list_v5: st.warning("Aucun arthropode n'a pu être identifié sur l'ensemble des images traitées.")
                else:
                    col_table_summary_display_v5, col_pie_chart_display_final_v5 = st.columns([2, 1]) 

                    with col_table_summary_display_v5:
                        st.subheader("Résumé des Identifications")
                        raw_label_counts_display_v5 = {}
                        for lbl_disp_v5 in all_labels_pie_tab2_an_list_v5:
                            raw_label_counts_display_v5[lbl_disp_v5] = raw_label_counts_display_v5.get(lbl_disp_v5, 0) + 1
                        
                        summary_data_display_v5 = []
                        for label_name_disp_v5, count_disp_v5 in sorted(raw_label_counts_display_v5.items(), key=lambda item_disp_v5: item_disp_v5[1], reverse=True):
                            display_label_name_val_v5 = "Opiliones et Araneae" if label_name_disp_v5 == "Arachnides" else label_name_disp_v5
                            
                            # MODIFICATION: Groupe taxonomique sans italique pour le tableau
                            term_to_display_in_table = display_label_name_val_v5 
                            
                            eco_func_disp_v5 = ECOLOGICAL_FUNCTIONS_MAP.get(display_label_name_val_v5, ECOLOGICAL_FUNCTIONS_MAP.get(label_name_disp_v5, DEFAULT_ECOLOGICAL_FUNCTION))
                            summary_data_display_v5.append({
                                "Groupe Taxonomique": term_to_display_in_table, 
                                "Quantité": count_disp_v5,
                                "Fonction Écologique": eco_func_disp_v5
                            })
                        
                        if summary_data_display_v5:
                            df_summary_display_v5 = pd.DataFrame(summary_data_display_v5)
                            
                            # MODIFICATION: Style du DataFrame pour affichage et téléchargement
                            styled_df = df_summary_display_v5.style.set_properties(**{
                                'font-size': '1.2em', # Police plus grande
                                'width': '100%'       # Utilise la largeur de la colonne
                            }).set_table_attributes('style="border-collapse: collapse; width: 100%;"') \
                              .set_table_styles([
                                  {'selector': 'th', 'props': [('border', '1px solid #ddd'), ('padding', '10px'), ('text-align', 'left'), ('background-color', '#f0f0f0')]},
                                  {'selector': 'td', 'props': [('border', '1px solid #ddd'), ('padding', '10px'), ('text-align', 'left')]}
                              ])
                            html_table = styled_df.to_html(escape=False, index=False, justify='left')
                            st.markdown(html_table, unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True) # Espace

                            # Bouton de téléchargement pour le tableau
                            csv_data_download = convert_df_to_csv(df_summary_display_v5) 
                            st.download_button(
                                label="📥 Télécharger le résumé (CSV)",
                                data=csv_data_download,
                                file_name="resume_identifications_arthropodes.csv",
                                mime="text/csv",
                                key="download_summary_csv_button",
                                use_container_width=True
                            )
                            st.markdown("<br>", unsafe_allow_html=True) # Espace

                        else:
                            st.write("Aucune donnée pour le tableau récapitulatif.")

                        # Calcul de l'indice de Shannon (reste identique)
                        ecological_counts_for_shannon_calc_v5 = {}
                        for data_row_shannon_v5 in summary_data_display_v5: # Utilise summary_data_display_v5 qui a les termes non-italiques
                            func_shannon_v5 = data_row_shannon_v5["Fonction Écologique"]
                            ecological_counts_for_shannon_calc_v5[func_shannon_v5] = ecological_counts_for_shannon_calc_v5.get(func_shannon_v5, 0) + data_row_shannon_v5["Quantité"]
                        
                        if ecological_counts_for_shannon_calc_v5:
                            shannon_val_display_v5 = calculate_shannon_index(ecological_counts_for_shannon_calc_v5)
                            st.metric(label="Indice de Shannon Fonctionnel Global (H')", value=f"{shannon_val_display_v5:.3f}")
                            if shannon_val_display_v5 == 0 and sum(ecological_counts_for_shannon_calc_v5.values()) > 0:
                                st.caption("H'=0: une seule fonction écologique présente.")
                            elif shannon_val_display_v5 > 0:
                                num_functions = len(ecological_counts_for_shannon_calc_v5)
                                max_s_disp_v5 = math.log(num_functions) if num_functions > 0 else 0
                                st.caption(f"Max H' pour {num_functions} fonction(s): {max_s_disp_v5:.3f}.")
                        else:
                            st.caption("Aucune donnée pour l'indice de Shannon.")

                    with col_pie_chart_display_final_v5:
                        ecological_counts_for_pie_chart_final_v5 = ecological_counts_for_shannon_calc_v5 
                        if ecological_counts_for_pie_chart_final_v5:
                            st.subheader("Fonctions Écologiques")
                            labels_pie_keys_final_v5 = list(ecological_counts_for_pie_chart_final_v5.keys())
                            sizes_pie_values_final_v5 = list(ecological_counts_for_pie_chart_final_v5.values())
                            colors_map_pie_final_v5 = {"Décomposeurs et Ingénieurs du sol": "#8B4513", "Pollinisateurs": "#FFD700", 
                                              "Prédateurs": "#DC143C", "Ravageur": "#FF8C00", "Non défini": "#D3D3D3"}
                            pie_colors_list_final_v5 = [colors_map_pie_final_v5.get(lbl_p_final_v5, "#CCCCCC") for lbl_p_final_v5 in labels_pie_keys_final_v5]
                            
                            fig_pie_final_display_v5, ax_pie_final_display_v5 = plt.subplots(figsize=(2.5, 1.75)) 
                            ax_pie_final_display_v5.pie(sizes_pie_values_final_v5, labels=None, autopct='%1.0f%%', startangle=90, 
                                       colors=pie_colors_list_final_v5, pctdistance=0.8, textprops={'fontsize': 4}) 
                            ax_pie_final_display_v5.axis('equal')
                            legend_handles_v5 = [plt.Rectangle((0,0),1,1, color=colors_map_pie_final_v5.get(name_v5, "#CCCCCC")) for name_v5 in labels_pie_keys_final_v5]
                            ax_pie_final_display_v5.legend(legend_handles_v5, labels_pie_keys_final_v5, loc='upper center', 
                                                    bbox_to_anchor=(0.5, -0.05), ncol=max(1, len(labels_pie_keys_final_v5)//2 + len(labels_pie_keys_final_v5)%2 ), 
                                                    fontsize='xx-small', frameon=False) 
                            plt.subplots_adjust(bottom=0.3 if len(labels_pie_keys_final_v5)>2 else 0.2) 
                            st.pyplot(fig_pie_final_display_v5)
                        else: st.write("Aucune fonction écologique à afficher.")
            
            st.markdown("--- \n ### Identification Détaillée par Image")
            for idx_detail_tab2_disp_final_v5, img_data_item_detail_id_tab2_disp_final_v5 in enumerate(imgs_processed_tab2_an_list_v5):
                st.markdown(f"#### {img_data_item_detail_id_tab2_disp_final_v5['filename']}")
                if not (img_data_item_detail_id_tab2_disp_final_v5.get("cv_image") is not None and 
                        img_data_item_detail_id_tab2_disp_final_v5.get("processed_data", {}).get("filtered_props") is not None and
                        img_data_item_detail_id_tab2_disp_final_v5.get("params", {}).get("margin") is not None):
                    st.write("Données de segmentation incomplètes pour cette image.")
                    continue
                extracted_arthropods_detail_id_tab2_disp_final_v5 = extract_arthropods(
                    img_data_item_detail_id_tab2_disp_final_v5["cv_image"], 
                    img_data_item_detail_id_tab2_disp_final_v5["processed_data"]["filtered_props"], 
                    img_data_item_detail_id_tab2_disp_final_v5["params"]["margin"]
                )
                if not extracted_arthropods_detail_id_tab2_disp_final_v5:
                    st.write("Aucun arthropode extrait pour identification sur cette image.")
                    continue
                num_cols_id_detail_disp_tab2_val_final_v5 = 3 
                cols_id_detail_disp_tab2_val_final_v5 = st.columns(num_cols_id_detail_disp_tab2_val_final_v5)
                col_idx_id_detail_disp_tab2_val_final_v5 = 0
                for arthropod_detail_item_id_tab2_disp_final_v5 in extracted_arthropods_detail_id_tab2_disp_final_v5:
                    label_detail_id_tab2_val_final_v5, confidence_detail_id_tab2_val_final_v5, _ = predict_arthropod_saved_model(
                        arthropod_detail_item_id_tab2_disp_final_v5["image"], model_to_use, class_names_to_use, MODEL_INPUT_SIZE
                    )
                    with cols_id_detail_disp_tab2_val_final_v5[col_idx_id_detail_disp_tab2_val_final_v5 % num_cols_id_detail_disp_tab2_val_final_v5]:
                        st.image(cv2.cvtColor(arthropod_detail_item_id_tab2_disp_final_v5["image"], cv2.COLOR_BGR2RGB), caption=f"Arthropode #{arthropod_detail_item_id_tab2_disp_final_v5['index'] + 1}", width=150)
                        if "Erreur" in label_detail_id_tab2_val_final_v5:
                            st.error(f"{label_detail_id_tab2_val_final_v5} ({confidence_detail_id_tab2_val_final_v5*100:.2f}%)")
                        else:
                            label_to_display_detail_val_final_v5 = "Opiliones et Araneae" if label_detail_id_tab2_val_final_v5 == "Arachnides" else label_detail_id_tab2_val_final_v5
                            
                            # Italicize for detailed display (kept as before)
                            term_to_display_final_detail = label_to_display_detail_val_final_v5
                            if label_to_display_detail_val_final_v5 in LATIN_TERMS_FOR_ITALICS or \
                               (label_to_display_detail_val_final_v5 == "Opiliones et Araneae" and ("Opiliones" in LATIN_TERMS_FOR_ITALICS or "Araneae" in LATIN_TERMS_FOR_ITALICS)):
                                term_to_display_final_detail = f"*{label_to_display_detail_val_final_v5}*"

                            st.markdown(f"**Label:** {term_to_display_final_detail}", unsafe_allow_html=True)
                            st.markdown(f"**Fonction:** {ECOLOGICAL_FUNCTIONS_MAP.get(label_to_display_detail_val_final_v5, ECOLOGICAL_FUNCTIONS_MAP.get(label_detail_id_tab2_val_final_v5, DEFAULT_ECOLOGICAL_FUNCTION))}")
                            st.markdown(f"**Confiance:** {confidence_detail_id_tab2_val_final_v5*100:.2f}%")
                    col_idx_id_detail_disp_tab2_val_final_v5 += 1
                if col_idx_id_detail_disp_tab2_val_final_v5 > 0 : st.markdown("---") 

    with tab3:
        st.header("Guide dʼutilisation")
        st.subheader("Segmentation par Image (Onglet 1)")
        st.write("""
        1.  **Téléversez vos images** en utilisant le bouton "1. Choisissez vos images". Vous pouvez sélectionner plusieurs fichiers.
        2.  Pour chaque image téléversée, une miniature s'affiche. Sous chaque image, un bouton "⚙️ Configurer" est disponible. Cliquez dessus pour **sélectionner cette image comme 'active'**.
        3.  Une fois une image sélectionnée, ses **paramètres de segmentation spécifiques s'affichent et peuvent être modifiés dans la barre latérale de gauche**. Le nom de l'image active est indiqué en haut de la barre latérale.
        4.  Après avoir ajusté les paramètres dans la barre latérale, cliquez sur le bouton **"Appliquer & Traiter Image Active"** (situé DANS la barre latérale) pour traiter UNIQUEMENT l'image actuellement sélectionnée avec ses nouveaux paramètres. Les résultats (image morphologique et détections) s'afficheront sous l'image originale dans l'onglet principal.
        5.  Un bouton général **"▶️ Segmenter les Images (avec leurs paramètres individuels)"** est disponible en haut de la liste des images dans l'onglet "Segmentation par Image". Cliquez dessus pour lancer la segmentation sur **TOUTES les images téléversées**, chacune utilisant ses propres paramètres (ceux par défaut ou ceux que vous avez ajustés individuellement).
        """)
        st.subheader("Analyse Globale (Onglet 2)")
        st.write("""
        Cet onglet devient actif une fois que des images ont été traitées (segmentées ET identifiées). Il affiche :
        -   Un **tableau récapitulatif** des groupes taxonomiques identifiés, leur quantité, et leur fonction écologique. Ce tableau peut être téléchargé au format CSV.
        -   Un **graphique circulaire (camembert)** illustrant la proportion des différentes fonctions écologiques.
        -   L'**Indice de Shannon Fonctionnel Global (H')**, calculé sur la base des fonctions écologiques.
        -   Une section **"Identification Détaillée par Image"** où chaque arthropode extrait de chaque image est montré avec son label d'identification, sa fonction écologique, et le score de confiance du modèle.
        Assurez-vous que le modèle d'identification et les labels sont correctement chargés (un message de succès apparaît dans la sidebar au démarrage si c'est le cas).
        """)
        st.subheader("Conseils")
        st.write("""
        - Si les résultats de segmentation ne sont pas satisfaisants pour une image, ajustez ses paramètres (flou, aire minimale, etc.) via la barre latérale et retraitez-la.
        - Le filtre de circularité et le filtre relatif peuvent aider à affiner la détection en éliminant les objets non pertinents. Expérimentez avec ces options.
        - Le modèle d'identification est entraîné sur un ensemble spécifique de classes. Les identifications pour des arthropodes très différents de ces classes peuvent être moins précises.
        """)


    st.markdown("---")
    ps_text = """
    **PS :** Quelques espèces de Carabidae (carabes) consomment des graines d'adventices, voire de semences agricoles (très marginalement). 
    Les Isopodes peuvent aussi consommer de jeunes pousses (rare et faible impact).
    En cas de photos d'arthropodes en dehors des classes définies par le modèle, l'outil renverra la classe qu'il considère comme étant la plus proche visuellement. 
    Ainsi, une photo d'Andrenidae pourrait être classée comme Apidae, bien que le modèle ait été entraîné sur des photos d'Apidae.
    """
    st.markdown(italicize_latin_terms(ps_text, LATIN_TERMS_FOR_ITALICS), unsafe_allow_html=True)


if __name__ == "__main__":
    if 'image_data_list' not in st.session_state: st.session_state.image_data_list = []
    if 'model_obj' not in st.session_state: st.session_state.model_obj = None
    if 'class_names_list' not in st.session_state: st.session_state.class_names_list = None
    if 'active_image_id_for_params' not in st.session_state: st.session_state.active_image_id_for_params = None
    if 'first_model_load_message_displayed' not in st.session_state: st.session_state.first_model_load_message_displayed = False
    main()
