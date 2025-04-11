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
