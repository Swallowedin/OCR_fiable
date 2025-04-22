import streamlit as st
import cv2
import numpy as np
import pytesseract
import os
import tempfile
import requests
import json
from PIL import Image
import io
from pdf2image import convert_from_bytes
import docx2txt
import sys

# Configuration de la page
st.set_page_config(
    page_title="OCR Amélioré pour Documents Commerciaux",
    page_icon="📝",
    layout="wide"
)

# Titre et description
st.title("OCR Amélioré pour Documents Commerciaux")
st.markdown("""
Cet outil extrait le texte des documents PDF, images et Word en utilisant des techniques avancées d'OCR.
Il combine plusieurs méthodes pour maximiser la qualité de l'extraction, même sur des documents de faible qualité.
""")

# Configuration de Tesseract pour Streamlit Cloud
if not os.environ.get('TESSDATA_PREFIX'):
    os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata'

# Fonction pour obtenir la clé API OpenAI
def get_openai_api_key():
    api_key = st.session_state.get("openai_api_key", "")
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except:
            api_key = ""
    return api_key

# Fonction pour obtenir la clé API OCR.space
def get_ocr_api_key():
    api_key = st.session_state.get("ocr_api_key", "")
    if not api_key:
        try:
            api_key = st.secrets["OCR_API_KEY"]
        except:
            # Clé par défaut (à remplacer idéalement par votre propre clé)
            api_key = "K88510884388957"
    return api_key

# Prétraitement d'image pour l'OCR
def preprocess_image_for_ocr(img):
    """Prétraitement avancé d'une image pour améliorer les résultats OCR."""
    # Gestion des erreurs robuste pour Streamlit Cloud
    try:
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Appliquer une légère réduction de bruit
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Augmenter le contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(denoised)
        
        # Binarisation avec Otsu pour une meilleure segmentation
        _, thresh = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Dilatation légère pour renforcer les caractères
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        return dilated
    except Exception as e:
        st.warning(f"Erreur de prétraitement d'image: {e}")
        return img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Fonction pour essayer plusieurs méthodes d'OCR
def extract_text_with_multiple_methods(img):
    """Essaie plusieurs méthodes d'OCR et retourne le meilleur résultat."""
    results = []
    
    # Méthode 1: Image originale
    try:
        text1 = pytesseract.image_to_string(img, lang='fra', config='--psm 6')
        results.append((text1, len(text1.strip())))
    except Exception as e:
        st.warning(f"Méthode OCR 1 non disponible: {e}")
    
    # Méthode 2: Prétraitement avancé
    try:
        processed = preprocess_image_for_ocr(img)
        text2 = pytesseract.image_to_string(processed, lang='fra', config='--psm 6')
        results.append((text2, len(text2.strip())))
    except Exception as e:
        st.warning(f"Méthode OCR 2 non disponible: {e}")
    
    # Méthode 3: Binarisation simple
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        text3 = pytesseract.image_to_string(binary, lang='fra', config='--psm 6')
        results.append((text3, len(text3.strip())))
    except Exception as e:
        st.warning(f"Méthode OCR 3 non disponible: {e}")
    
    # Méthode 4: Orientation spécifique pour les tableaux
    try:
        text4 = pytesseract.image_to_string(img, lang='fra', config='--psm 4')
        results.append((text4, len(text4.strip())))
    except Exception as e:
        st.warning(f"Méthode OCR 4 non disponible: {e}")
    
    # Retourner le résultat avec le plus de texte
    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        return results[0][0]
    
    return ""

# Fonction pour extraire le texte d'une image
def extract_text_from_image(image_data):
    """Extrait le texte d'une image avec OCR amélioré."""
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Utiliser la méthode multiple
        return extract_text_with_multiple_methods(img)
    
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du texte de l'image: {str(e)}")
        # Tentative de secours avec l'API OCR
        try:
            return ocr_from_image_using_api(image_data)
        except Exception as e:
            st.error(f"API OCR.space échouée également: {e}")
            return ""

# Fonction pour extraire le texte d'un PDF
def extract_text_from_pdf(pdf_data):
    """Extrait le texte d'un fichier PDF avec une approche hybride."""
    import PyPDF2
    
    text = ""
    
    # Essayer d'abord l'extraction native de PyPDF2
    try:
        with io.BytesIO(pdf_data) as pdf_stream:
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            for page_num in range(len(pdf_reader.pages)):
                page_text = pdf_reader.pages[page_num].extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.warning(f"Extraction native du PDF non réussie: {str(e)}")
    
    # Si peu ou pas de texte extrait, essayer l'OCR
    if len(text.strip()) < 100:  # Seuil arbitraire pour déterminer si l'extraction est insuffisante
        st.info("Extraction de texte limitée, utilisation de l'OCR...")
        
        try:
            # Convertir le PDF en images
            images = convert_from_bytes(pdf_data, dpi=200)  # Résolution réduite pour Streamlit Cloud
            
            # OCR sur chaque page
            page_texts = []
            for i, img in enumerate(images):
                with st.spinner(f"OCR sur la page {i+1}/{len(images)}..."):
                    # Convertir PIL Image en format OpenCV
                    open_cv_image = np.array(img)
                    # RGB vers BGR (format OpenCV)
                    open_cv_image = open_cv_image[:, :, ::-1].copy()
                    
                    # Extraire le texte avec notre méthode multiple
                    page_text = extract_text_with_multiple_methods(open_cv_image)
                    page_texts.append(page_text)
            
            # Si on a des résultats d'OCR, les utiliser
            if any(page_texts):
                text = "\n\n".join(page_texts)
            # Sinon, dernier recours: API OCR externe
            else:
                text = ocr_from_pdf_using_api(pdf_data)
                
        except Exception as e:
            st.error(f"Erreur lors de l'OCR du PDF: {str(e)}")
            # Dernier recours: API OCR externe
            try:
                text = ocr_from_pdf_using_api(pdf_data)
            except Exception as e:
                st.error(f"API OCR.space échouée également: {e}")
    
    return text

# Fonction pour utiliser l'API OCR.space sur une image
def ocr_from_image_using_api(image_data):
    """Utilise l'API OCR.space pour extraire le texte d'une image."""
    try:
        OCR_API_KEY = get_ocr_api_key()
        
        with io.BytesIO(image_data) as image_stream:
            files = {"file": image_stream}
            response = requests.post(
                "https://api.ocr.space/parse/image",
                files=files,
                data={
                    'apikey': OCR_API_KEY,
                    'language': 'fre',
                    'isTable': True,
                    'OCREngine': 2  # Utiliser le moteur OCR le plus précis
                }
            )
        
        result = response.json()
        
        if result["OCRExitCode"] == 1:
            return result['ParsedResults'][0]['ParsedText']
        else:
            st.error("Erreur dans le traitement OCR API: " + result.get("ErrorMessage", "Erreur inconnue"))
            return ""
    
    except Exception as e:
        st.error(f"Erreur lors de l'utilisation de l'API OCR: {str(e)}")
        return ""

# Fonction pour utiliser l'API OCR.space sur un PDF
def ocr_from_pdf_using_api(pdf_data):
    """Extraire le texte d'un PDF à l'aide de l'API OCR.Space avec options améliorées."""
    try:
        OCR_API_KEY = get_ocr_api_key()
        
        # Sauvegarder le fichier uploadé sur le système de fichiers temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(pdf_data)
            temp_pdf_path = tmp.name
        
        with open(temp_pdf_path, 'rb') as file:
            files = {'file': file}
            response = requests.post(
                "https://api.ocr.space/parse/image",
                files=files,
                data={
                    'apikey': OCR_API_KEY,
                    'language': 'fre',
                    'isTable': True,
                    'OCREngine': 2,  # Moteur plus précis
                    'scale': True,   # Redimensionnement automatique
                    'detectOrientation': True
                }
            )

        # Nettoyer le fichier temporaire
        try:
            os.unlink(temp_pdf_path)
        except:
            pass

        result = response.json()

        if result["OCRExitCode"] == 1:
            return result['ParsedResults'][0]['ParsedText']
        else:
            st.error("Erreur dans le traitement OCR API: " + result.get("ErrorMessage", "Erreur inconnue"))
            return ""
    
    except Exception as e:
        st.error(f"Erreur lors de l'OCR du PDF via API: {str(e)}")
        return ""

# Fonction pour extraire le texte d'un fichier Word
def extract_text_from_docx(docx_data):
    """Extraire le texte d'un fichier Word."""
    try:
        with io.BytesIO(docx_data) as docx_stream:
            text = docx2txt.process(docx_stream)
            return text
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du texte du fichier Word: {str(e)}")
        return ""

# Fonction pour traiter un fichier avec plusieurs méthodes de secours
def process_file(file):
    """Traite un fichier avec plusieurs méthodes de secours."""
    file_type = file.type
    file_data = file.getvalue()
    
    st.write(f"Type de fichier: {file_type}")
    
    # Première tentative: méthode standard selon le type de fichier
    if file_type == "application/pdf":
        text = extract_text_from_pdf(file_data)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(file_data)
    elif file_type == "text/plain":
        text = file_data.decode('utf-8')
    elif file_type.startswith("image/"):
        text = extract_text_from_image(file_data)
    else:
        st.warning(f"Type de fichier non pris en charge: {file_type}. Tentative d'OCR générique.")
        # Tenter OCR générique pour les types non reconnus
        text = ocr_from_image_using_api(file_data)
    
    # Si le texte est vide ou très court, essayer l'API OCR comme secours final
    if len(text.strip()) < 50:
        st.warning("Extraction de texte insuffisante. Tentative avec API OCR.")
        if file_type == "application/pdf":
            backup_text = ocr_from_pdf_using_api(file_data)
        else:
            backup_text = ocr_from_image_using_api(file_data)
        
        # N'utiliser le texte de secours que s'il est meilleur
        if len(backup_text.strip()) > len(text.strip()):
            text = backup_text
    
    return text

# Fonction pour améliorer le texte avec GPT-4o-mini
def enhance_text_with_gpt(text):
    """Améliore le texte extrait en utilisant GPT-4o-mini pour corriger les erreurs d'OCR."""
    api_key = get_openai_api_key()
    
    if not api_key:
        st.error("Clé API OpenAI manquante. Impossible d'utiliser GPT-4o-mini.")
        return text
    
    try:
        prompt = f"""
        Tu es un expert en correction de textes issus d'OCR. Le texte suivant provient d'une reconnaissance
        optique de caractères qui contient probablement des erreurs. Corrige les erreurs évidentes
        (caractères mal reconnus, mots tronqués, etc.) tout en préservant la structure et le formatage
        original du document.
        
        IMPORTANT:
        - Ne modifie que les erreurs manifestes d'OCR
        - Préserve la disposition des paragraphes et la structure du document
        - Ne complète pas les parties manquantes par tes propres mots
        - Ne reformule pas le texte, corrige uniquement les erreurs de reconnaissance
        
        Texte à corriger:
        ```
        {text}
        ```
        
        Retourne uniquement le texte corrigé, sans commentaires ni explications.
        """
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            enhanced_text = response.json()["choices"][0]["message"]["content"]
            return enhanced_text
        else:
            st.error(f"Erreur API OpenAI ({response.status_code}): {response.text}")
            return text
    
    except Exception as e:
        st.error(f"Erreur lors de l'amélioration du texte avec GPT: {str(e)}")
        return text

# Sidebar pour les configurations
with st.sidebar:
    st.header("Configuration")
    
    # Paramètres OCR
    st.subheader("Paramètres OCR")
    
    # OCR.space API Key
    ocr_api_key = st.text_input(
        "Clé API OCR.space (optionnel)",
        value=get_ocr_api_key(),
        type="password",
        help="Utilisée pour l'OCR de secours via API"
    )
    if ocr_api_key:
        st.session_state["ocr_api_key"] = ocr_api_key
    
    # OpenAI API Key
    st.subheader("Amélioration avec GPT-4o-mini")
    openai_api_key = st.text_input(
        "Clé API OpenAI (pour GPT-4o-mini)",
        value=get_openai_api_key(),
        type="password",
        help="Utilisée pour améliorer les résultats d'OCR"
    )
    if openai_api_key:
        st.session_state["openai_api_key"] = openai_api_key

    use_gpt = st.checkbox(
        "Utiliser GPT-4o-mini pour améliorer les résultats",
        value=True,
        help="Utilise l'IA pour corriger les erreurs d'OCR et améliorer le texte"
    )

# Interface principale
uploaded_file = st.file_uploader(
    "Téléchargez un document (PDF, image, docx)",
    type=["pdf", "png", "jpg", "jpeg", "docx", "txt"]
)

if uploaded_file:
    st.success(f"Fichier téléchargé: {uploaded_file.name}")
    
    # Extraction du texte
    with st.spinner("Extraction du texte en cours..."):
        extracted_text = process_file(uploaded_file)
    
    # Amélioration avec GPT-4o-mini si activé
    if use_gpt and get_openai_api_key():
        with st.spinner("Amélioration du texte avec GPT-4o-mini..."):
            enhanced_text = enhance_text_with_gpt(extracted_text)
    else:
        enhanced_text = extracted_text
    
    # Affichage des résultats
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Texte extrait brut")
        st.text_area("", extracted_text, height=400)
        st.download_button(
            "Télécharger le texte brut",
            extracted_text,
            file_name=f"{uploaded_file.name}_extrait.txt",
            mime="text/plain"
        )
    
    with col2:
        st.subheader("Texte amélioré" if use_gpt and get_openai_api_key() else "Texte extrait")
        st.text_area("", enhanced_text, height=400)
        st.download_button(
            "Télécharger le texte final",
            enhanced_text,
            file_name=f"{uploaded_file.name}_final.txt",
            mime="text/plain"
        )
    
    # Statistiques
    st.subheader("Statistiques")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Caractères extraits", len(extracted_text))
    
    with col2:
        st.metric("Mots extraits", len(extracted_text.split()))
    
    with col3:
        if use_gpt and get_openai_api_key():
            improvement = (len(enhanced_text) - len(extracted_text)) / len(extracted_text) * 100 if len(extracted_text) > 0 else 0
            st.metric("Amélioration", f"{improvement:.1f}%")
    
    # Affichage de l'aperçu du document (si image)
    if uploaded_file.type.startswith("image/"):
        st.subheader("Aperçu du document")
        st.image(uploaded_file, caption="Document original")
    
    # Si PDF, afficher la première page
    elif uploaded_file.type == "application/pdf":
        try:
            st.subheader("Aperçu de la première page")
            images = convert_from_bytes(uploaded_file.getvalue(), first_page=1, last_page=1)
            if images:
                st.image(images[0], caption="Première page du PDF")
        except Exception as e:
            st.error(f"Impossible d'afficher l'aperçu du PDF: {e}")

else:
    st.info("Veuillez télécharger un document pour commencer l'extraction de texte.")

# Footer
st.markdown("---")
st.markdown("""
**Note:** Cet outil utilise plusieurs méthodes d'OCR pour maximiser la qualité de l'extraction:
1. OCR natif avec prétraitements avancés d'image
2. API OCR.space comme solution de secours
3. Amélioration optionnelle avec GPT-4o-mini pour corriger les erreurs d'OCR
""")

# Affichage des informations environnementales pour débogage
if st.sidebar.checkbox("Afficher les informations de débogage", False):
    st.sidebar.subheader("Informations système")
    st.sidebar.text(f"Python version: {sys.version}")
    st.sidebar.text(f"OpenCV version: {cv2.__version__}")
    st.sidebar.text(f"Tesseract disponible: {'Oui' if pytesseract.get_tesseract_version() else 'Non'}")
    st.sidebar.text(f"TESSDATA_PREFIX: {os.environ.get('TESSDATA_PREFIX', 'Non défini')}")
