# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
import time

# Try to load class names from file, otherwise use default
def load_class_names():
    """Load class names from file or use defaults"""
    if os.path.exists('class_names.txt'):
        with open('class_names.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return tuple(class_names)
    else:
        # Updated default class names - all tomato diseases
        return (
            'Tomato-Bacterial_spot',
            'Tomato-Early_blight',
            'Tomato-healthy',
            'Tomato-Late_blight',
            'Tomato-Leaf_Mold',
            'Tomato-powdery_mildew',
            'Tomato-Septoria_leaf_spot',
            'Tomato-Spider_mites_Two-spotted_spider_mite',
            'Tomato-Target_Spot',
            'Tomato-Tomato_mosaic_virus',
            'Tomato-Tomato_Yellow_Leaf_Curl_Virus'
        )

# Loading the Model
try:
    model = load_model('plant_disease_model.h5')
    CLASS_NAMES = load_class_names()
    # Suppress success message if not the first run
    if 'model_loaded' not in st.session_state:
        st.success("✅ Model loaded successfully!")
        st.session_state.model_loaded = True
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.info("Please run the training script first to generate the model.")
    st.stop()
                    
# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0  # 0: upload, 1: validated, 2: diagnosed, 3: questions_answered, 4: plan_generated
if 'diagnosis' not in st.session_state:
    st.session_state.diagnosis = None
if 'is_tomato' not in st.session_state:
    st.session_state.is_tomato = False
if 'contextual_answers' not in st.session_state:
    st.session_state.contextual_answers = {}
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.0

# Function to generate action plan
def generate_action_plan(disease, answers):
    """Generate a comprehensive action plan based on diagnosis and contextual answers"""
    
    climate = answers.get('climate', '').lower()
    watering = answers.get('watering', '').lower()
    spread = answers.get('spread', '').lower()
    
    # Determine if conditions are wet/humid or dry
    is_wet = any(term in climate for term in ['wet', 'humid', 'rain', 'poor drainage', 'damp'])
    is_dry = any(term in climate for term in ['dry', 'arid'])
    
    # Determine watering method
    overhead = any(term in watering for term in ['overhead', 'sprinkler', 'sprinklers'])
    base_watering = any(term in watering for term in ['base', 'drip', 'soil'])
    
    # Determine spread
    isolated = any(term in spread for term in ['just this', 'only this', 'one', 'single'])
    widespread = any(term in spread for term in ['other', 'multiple', 'nearby', 'several', 'many', 'most/all'])
    
    plan = {
        "immediate_actions": [],
        "organic_remedies": [],
        "chemical_treatment": "",
        "long_term_prevention": []
    }
    
    disease_lower = disease.lower()
    
    # Healthy plant
    if 'healthy' in disease_lower:
        plan["immediate_actions"] = [
            "Continue monitoring your plant regularly for any signs of disease.",
            "Maintain current good practices - your plant appears healthy!"
        ]
        plan["organic_remedies"] = [
            "Continue preventive measures: proper spacing, good air circulation, and consistent watering.",
            "Apply organic fertilizers to maintain plant vigor.",
            "Use mulch to prevent soil splash-up and maintain soil moisture.",
            "Monitor for pests and diseases regularly."
        ]
        plan["chemical_treatment"] = "No chemical treatment needed for healthy plants. Continue with preventive organic practices."
        plan["long_term_prevention"] = [
            "Continue good cultural practices: proper spacing, crop rotation, and sanitation.",
            "Maintain healthy soil with organic matter and proper pH.",
            "Water consistently at the base of plants.",
            "Remove any dead or diseased plant material promptly."
        ]
    
    # Bacterial Spot
    elif 'bacterial_spot' in disease_lower:
        plan["immediate_actions"] = [
            "Immediately prune and destroy all affected leaves using sterilized pruning shears (wipe with 70% alcohol between cuts).",
            "Isolate this plant if possible, or create a barrier to prevent water splashing onto healthy plants.",
            "Stop overhead watering immediately if you're using it - switch to base watering only.",
            "Dispose of infected plant material in sealed bags; do not compost it."
        ]
        plan["organic_remedies"] = [
            "Apply a copper-based organic fungicide (like copper soap or Bordeaux mixture) every 7-10 days, especially during humid weather.",
            "Use a neem oil solution (2-3 tablespoons per gallon of water) as a foliar spray every 7-14 days to help suppress bacterial growth.",
            "Apply baking soda solution (1 tablespoon per gallon of water) with a few drops of liquid soap as a preventive measure.",
            "Improve air circulation by staking plants and ensuring proper spacing (at least 2-3 feet between plants).",
            f"{'Reduce watering frequency and ensure soil drainage is excellent.' if is_wet else 'Maintain consistent watering at the base of plants, avoiding wetting the leaves.'}",
            "Remove lower leaves that touch the soil to prevent soil splash-up contamination."
        ]
        plan["chemical_treatment"] = "If the infection is widespread and organic methods aren't controlling it, consider a copper-based bactericide (like copper hydroxide or copper oxychloride) applied every 7-10 days. For severe cases, products containing mancozeb or streptomycin may be necessary, but use these as a last resort and follow label instructions carefully."
        plan["long_term_prevention"] = [
            "Practice crop rotation - don't plant tomatoes in the same spot for at least 3 years.",
            "Use disease-resistant tomato varieties when possible (look for 'BS' or 'Bacterial Spot resistant' on seed packets).",
            "Ensure 2-3 feet of spacing between plants for adequate air circulation.",
            "Use mulch (straw or plastic) to prevent soil splash-up onto leaves.",
            "Water at the base of plants using drip irrigation or soaker hoses, never overhead.",
            "Sanitize all garden tools between uses and between seasons.",
            "Avoid working in the garden when plants are wet to prevent spreading bacteria.",
            "Start with disease-free seeds or transplants from reputable sources."
        ]
    
    # Early Blight
    elif 'early_blight' in disease_lower:
        plan["immediate_actions"] = [
            "Immediately remove and destroy all affected leaves and stems. Place them in sealed bags - do not compost.",
            "Prune lower leaves that are within 6-12 inches of the soil to prevent soil-borne spores from splashing up.",
            "Stop overhead watering immediately and switch to base/drip irrigation.",
            "If infection is severe, consider removing the entire plant to protect others."
        ]
        plan["organic_remedies"] = [
            "Apply neem oil solution (2-3 tablespoons per gallon) every 7-10 days, making sure to cover both sides of leaves.",
            "Use a baking soda spray (1 tablespoon baking soda, 1 teaspoon liquid soap, 1 gallon water) every 7-14 days.",
            "Apply compost tea or beneficial microorganisms to boost plant immunity.",
            "Ensure proper staking and pruning to improve air circulation around plants.",
            f"{'Improve drainage immediately - consider raised beds or adding organic matter to improve soil structure.' if is_wet else 'Maintain consistent, deep watering at the base to prevent stress.'}",
            "Apply organic mulch to prevent soil splash-up and maintain even soil moisture."
        ]
        plan["chemical_treatment"] = "For severe or rapidly spreading infections, apply a chlorothalonil-based fungicide (like Daconil) or a copper-based fungicide every 7-10 days. Mancozeb is also effective but requires careful application. Always follow label instructions and observe pre-harvest intervals."
        plan["long_term_prevention"] = [
            "Practice strict crop rotation - avoid planting tomatoes, potatoes, or peppers in the same area for 3-4 years.",
            "Remove all plant debris at the end of the season and destroy it (don't compost).",
            "Use certified disease-free seeds and transplants.",
            "Ensure proper spacing (2-3 feet) and stake plants for better air circulation.",
            "Use mulch to prevent soil splash-up and maintain even soil moisture.",
            "Water at the base only, preferably in the morning so leaves dry quickly.",
            "Choose resistant varieties when available (look for 'EB' or 'Early Blight resistant')."
        ]
    
    # Late Blight
    elif 'late_blight' in disease_lower:
        plan["immediate_actions"] = [
            "URGENT: Late blight spreads rapidly! Immediately remove and destroy ALL affected plant parts in sealed bags.",
            "If more than 50% of the plant is affected, remove the entire plant immediately to save others.",
            "Stop all overhead watering immediately.",
            "Do not compost infected material - dispose of it away from your garden."
        ]
        plan["organic_remedies"] = [
            "Apply copper-based fungicides immediately (copper sulfate or Bordeaux mixture) every 5-7 days during wet weather.",
            "Use neem oil or compost tea as supplementary treatments, but these alone are usually insufficient for late blight.",
            "Improve air circulation dramatically - space plants 3-4 feet apart and use cages or stakes.",
            "Remove all lower leaves within 12-18 inches of the ground.",
            "Apply thick mulch to prevent soil splash-up."
        ]
        plan["chemical_treatment"] = "Late blight requires immediate chemical intervention. Apply chlorothalonil (Daconil), mancozeb, or fosetyl-aluminum-based fungicides every 5-7 days during wet conditions. Rotate between different fungicide classes to prevent resistance. This is critical - late blight can destroy entire crops in days."
        plan["long_term_prevention"] = [
            "Practice strict 3-4 year crop rotation away from tomatoes, potatoes, and peppers.",
            "Destroy ALL plant debris at season end - do not compost.",
            "Choose late blight-resistant varieties (look for 'LB' resistance markers).",
            "Ensure excellent drainage and avoid planting in low-lying areas.",
            "Water only at the base, preferably in the morning.",
            "Monitor weather forecasts and apply preventive fungicides before extended wet periods.",
            "Space plants widely (3-4 feet) for maximum air circulation."
        ]
    
    # Leaf Mold
    elif 'leaf_mold' in disease_lower:
        plan["immediate_actions"] = [
            "Remove and destroy all affected leaves immediately.",
            "Improve air circulation by pruning and spacing plants wider.",
            "Reduce humidity around plants - avoid overhead watering.",
            "Remove lower leaves that are heavily infected."
        ]
        plan["organic_remedies"] = [
            "Apply sulfur-based fungicides or neem oil every 7-10 days.",
            "Improve air circulation through proper spacing (3 feet minimum) and pruning.",
            "Use fans in greenhouse settings to reduce humidity.",
            "Apply compost tea to boost plant immunity.",
            f"{'Reduce watering frequency and improve ventilation immediately.' if is_wet else 'Maintain consistent base watering while ensuring good air flow.'}",
            "Ensure plants receive adequate sunlight to reduce leaf wetness."
        ]
        plan["chemical_treatment"] = "If organic methods fail, apply chlorothalonil, copper-based fungicides, or mancozeb every 7-10 days. In greenhouse settings, consider systemic fungicides. Always follow label instructions."
        plan["long_term_prevention"] = [
            "Choose leaf mold-resistant varieties (look for 'LM' resistance).",
            "Ensure excellent air circulation - space plants 3 feet apart minimum.",
            "Water at the base only, never overhead.",
            "In greenhouses, use fans and proper ventilation to keep humidity below 85%.",
            "Remove all plant debris at season end.",
            "Practice crop rotation.",
            "Avoid working with plants when they are wet."
        ]
    
    # Powdery Mildew
    elif 'powdery_mildew' in disease_lower:
        plan["immediate_actions"] = [
            "Remove and destroy heavily infected leaves immediately.",
            "Improve air circulation by pruning and spacing.",
            "Reduce humidity around plants.",
            "Apply treatment as soon as symptoms appear."
        ]
        plan["organic_remedies"] = [
            "Apply sulfur-based fungicides or potassium bicarbonate solution (1 tablespoon per gallon) every 7-10 days.",
            "Use neem oil or horticultural oil sprays every 7-14 days.",
            "Apply milk solution (1 part milk to 9 parts water) as a preventive measure.",
            "Improve air circulation through proper spacing and pruning.",
            "Ensure plants receive adequate sunlight.",
            "Water at the base only, avoiding wetting leaves."
        ]
        plan["chemical_treatment"] = "For severe infections, apply fungicides containing myclobutanil, propiconazole, or triflumizole. Systemic fungicides can be effective but should be rotated to prevent resistance. Follow label instructions carefully."
        plan["long_term_prevention"] = [
            "Choose powdery mildew-resistant varieties when available.",
            "Maintain proper spacing (2-3 feet) for good air circulation.",
            "Water at the base only, preferably in the morning.",
            "Prune plants to improve airflow.",
            "Avoid excessive nitrogen fertilization which promotes susceptible new growth.",
            "Remove and destroy all plant debris at season end.",
            "Practice crop rotation."
        ]
    
    # Septoria Leaf Spot
    elif 'septoria' in disease_lower:
        plan["immediate_actions"] = [
            "Remove and destroy all affected leaves immediately - this disease spreads through water splash.",
            "Prune lower leaves within 6-12 inches of soil.",
            "Stop overhead watering immediately.",
            "Dispose of infected material in sealed bags."
        ]
        plan["organic_remedies"] = [
            "Apply copper-based fungicides or sulfur every 7-10 days.",
            "Use neem oil solution every 7-14 days.",
            "Apply baking soda spray (1 tablespoon per gallon) as preventive treatment.",
            "Improve air circulation through proper spacing and staking.",
            f"{'Improve drainage and reduce watering frequency.' if is_wet else 'Maintain consistent base watering.'}",
            "Apply thick mulch to prevent soil splash-up.",
            "Remove lower leaves regularly to prevent soil contact."
        ]
        plan["chemical_treatment"] = "For severe cases, apply chlorothalonil (Daconil), mancozeb, or copper-based fungicides every 7-10 days. Start early and continue through the season, especially during wet periods. Rotate fungicides to prevent resistance."
        plan["long_term_prevention"] = [
            "Practice 3-year crop rotation away from tomatoes and related crops.",
            "Remove and destroy ALL plant debris at season end.",
            "Use mulch to prevent soil splash-up.",
            "Water at the base only, never overhead.",
            "Space plants 2-3 feet apart for good air circulation.",
            "Choose resistant varieties when available.",
            "Sanitize garden tools between uses."
        ]
    
    # Spider Mites
    elif 'spider_mites' in disease_lower or 'mite' in disease_lower:
        plan["immediate_actions"] = [
            "Isolate the affected plant if possible to prevent mite spread.",
            "Prune and remove heavily infested leaves.",
            "Increase humidity around plants (mites thrive in dry conditions).",
            "Apply treatment immediately - mites reproduce rapidly."
        ]
        plan["organic_remedies"] = [
            "Spray plants with a strong stream of water to dislodge mites (especially underside of leaves).",
            "Apply neem oil, insecticidal soap, or horticultural oil every 3-5 days for 2-3 weeks.",
            "Introduce beneficial insects like ladybugs or predatory mites if available.",
            "Increase humidity by misting plants or using a humidifier.",
            "Apply diatomaceous earth around plants (wear a mask when applying).",
            "Use rosemary oil or peppermint oil sprays as natural repellents."
        ]
        plan["chemical_treatment"] = "For severe infestations, use miticides containing abamectin, bifenazate, or spiromesifen. Rotate between different active ingredients to prevent resistance. Apply to both tops and undersides of leaves. Follow label instructions carefully."
        plan["long_term_prevention"] = [
            "Maintain adequate humidity levels (spider mites prefer dry conditions).",
            "Regularly inspect plants, especially during hot, dry weather.",
            "Keep plants well-watered and healthy (stressed plants are more susceptible).",
            "Introduce beneficial insects early in the season.",
            "Remove weeds around the garden that can harbor mites.",
            "Avoid excessive use of broad-spectrum insecticides that kill beneficial insects.",
            "Practice good garden hygiene and remove plant debris."
        ]
    
    # Target Spot
    elif 'target_spot' in disease_lower:
        plan["immediate_actions"] = [
            "Remove and destroy affected leaves immediately.",
            "Improve air circulation through spacing and pruning.",
            "Stop overhead watering.",
            "Remove lower leaves that touch the soil."
        ]
        plan["organic_remedies"] = [
            "Apply copper-based fungicides every 7-10 days.",
            "Use neem oil or sulfur-based treatments.",
            "Apply baking soda solution as a preventive measure.",
            "Improve air circulation through proper spacing (2-3 feet).",
            "Water at the base only.",
            "Apply mulch to prevent soil splash-up.",
            f"{'Improve drainage and reduce humidity.' if is_wet else 'Maintain consistent base watering.'}"
        ]
        plan["chemical_treatment"] = "For severe infections, apply chlorothalonil, mancozeb, or azoxystrobin-based fungicides every 7-10 days. Start treatment early and continue through the season, especially during warm, humid weather."
        plan["long_term_prevention"] = [
            "Practice crop rotation (3-year cycle).",
            "Remove and destroy all plant debris at season end.",
            "Use mulch to prevent soil splash-up.",
            "Water at the base only, preferably in the morning.",
            "Maintain proper spacing (2-3 feet) for air circulation.",
            "Choose resistant varieties when available.",
            "Sanitize garden tools regularly."
        ]
    
    # Mosaic Virus
    elif 'mosaic_virus' in disease_lower:
        plan["immediate_actions"] = [
            "URGENT: Remove and destroy entire infected plants immediately - viruses cannot be cured.",
            "Do not compost infected plants - dispose of them away from the garden.",
            "Sanitize all tools that touched infected plants.",
            "Check other plants for symptoms and remove them if found."
        ]
        plan["organic_remedies"] = [
            "There is no cure for viral diseases - prevention is the only solution.",
            "Control aphids and whiteflies (virus vectors) using neem oil, insecticidal soap, or beneficial insects.",
            "Remove weeds that can harbor viruses and insect vectors.",
            "Use row covers to prevent insect vectors from reaching plants.",
            "Maintain plant health through proper nutrition and watering."
        ]
        plan["chemical_treatment"] = "There are no effective chemical treatments for viral diseases. Focus on controlling insect vectors (aphids, whiteflies) with insecticides like pyrethroids or neonicotinoids. However, prevention through resistant varieties is far more effective."
        plan["long_term_prevention"] = [
            "Choose virus-resistant tomato varieties (look for 'TMV', 'ToMV', or 'virus-resistant' on seed packets).",
            "Control insect vectors (aphids, whiteflies, thrips) throughout the season.",
            "Start with certified virus-free seeds and transplants.",
            "Sanitize all garden tools between plants and seasons.",
            "Remove weeds that can serve as virus reservoirs.",
            "Use row covers or netting to exclude insect vectors.",
            "Avoid working with plants when they are wet.",
            "Do not smoke around tomato plants (tobacco can carry mosaic virus)."
        ]
    
    # Yellow Leaf Curl Virus
    elif 'yellow_leaf_curl' in disease_lower or 'leaf_curl' in disease_lower:
        plan["immediate_actions"] = [
            "URGENT: Remove and destroy entire infected plants immediately.",
            "Control whiteflies immediately - they are the primary vector.",
            "Do not compost infected plants.",
            "Check surrounding plants and remove any showing symptoms."
        ]
        plan["organic_remedies"] = [
            "Control whiteflies using yellow sticky traps, neem oil, or insecticidal soap.",
            "Introduce beneficial insects like Encarsia formosa (parasitic wasp) to control whiteflies.",
            "Use reflective mulches to repel whiteflies.",
            "Apply row covers to prevent whitefly access to plants.",
            "Remove weeds that can harbor whiteflies and the virus.",
            "There is no cure for infected plants - they must be removed."
        ]
        plan["chemical_treatment"] = "Control whitefly populations using insecticides like pyrethroids, neonicotinoids, or insect growth regulators. However, once plants are infected, they cannot be cured. Focus on preventing whitefly infestations and using resistant varieties."
        plan["long_term_prevention"] = [
            "Choose TYLCV-resistant varieties (look for 'TYLCV-resistant' or 'TY' on seed packets).",
            "Control whiteflies aggressively throughout the season using multiple methods.",
            "Use reflective mulches or row covers to exclude whiteflies.",
            "Remove weeds that serve as whitefly hosts.",
            "Start with certified virus-free transplants.",
            "Practice good garden sanitation - remove plant debris immediately.",
            "Consider growing in screened greenhouses in areas with high whitefly pressure.",
            "Monitor for whiteflies regularly and take action immediately if detected."
        ]
    
    # Generic plan for unrecognized diseases
    else:
        plan["immediate_actions"] = [
            "Immediately isolate affected plants if possible.",
            "Prune and remove all visibly infected leaves and plant parts.",
            "Dispose of infected material in sealed bags - do not compost.",
            "Stop any overhead watering immediately."
        ]
        plan["organic_remedies"] = [
            "Apply neem oil solution (2-3 tablespoons per gallon) every 7-10 days.",
            "Use organic fungicides like copper-based products or sulfur-based treatments.",
            "Improve air circulation through proper spacing and staking.",
            f"{'Improve drainage and reduce watering frequency.' if is_wet else 'Maintain consistent, deep watering at the base.'}",
            "Apply organic mulch to prevent soil splash-up.",
            "Remove lower leaves that touch the soil."
        ]
        plan["chemical_treatment"] = "If organic methods are ineffective, consider copper-based fungicides or other appropriate chemical controls based on the specific disease. Always follow label instructions and safety guidelines. Consult with a local extension service for disease-specific recommendations."
        plan["long_term_prevention"] = [
            "Practice crop rotation (3-4 year cycles).",
            "Use disease-resistant varieties when available.",
            "Maintain proper plant spacing (2-3 feet) for air circulation.",
            "Water at the base of plants using drip irrigation.",
            "Use mulch to prevent soil splash-up.",
            "Sanitize garden tools regularly.",
            "Remove and destroy all plant debris at season end.",
            "Start with disease-free seeds and transplants."
        ]
    
    # Adjust based on spread
    if widespread and not isolated:
        plan["immediate_actions"].insert(0, "⚠️ WARNING: This appears to be affecting multiple plants. Take immediate action to prevent further spread.")
        plan["chemical_treatment"] = "Given that this is affecting multiple plants, chemical treatment may be necessary to prevent total crop loss. " + plan["chemical_treatment"]
    
    return plan

# Setting Title of App
st.title("🍅 Agri-Bot: Expert Tomato Plant Pathologist")
st.markdown("**Your AI agronomist specializing in tomato cultivation**")
st.markdown("---")

# STEP 1: IMAGE VALIDATION
st.header("Step 1: Image Upload & Validation")

plant_image = st.file_uploader("Choose an image of your tomato leaf...", type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"])
submit = st.button('🔍 Analyze Image')

if submit:
    if plant_image is not None:
        # Convert the file to an opencv image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        if opencv_image is None:
            st.error("❌ **Error loading image. Please ensure the file is a valid image format.**")
        else:
            # Store the image
            st.session_state.uploaded_image = opencv_image.copy()
            
            # Displaying the image
            st.image(opencv_image, channels="BGR", caption="Uploaded Image")
            
            # Resize image to match model input size (224x224 as per training)
            resized_image = cv2.resize(opencv_image, (224, 224))
            
            # Convert BGR to RGB (OpenCV uses BGR, but model expects RGB)
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            
            # Apply EfficientNet preprocessing (matches training preprocessing)
            # preprocess_input handles normalization to match ImageNet statistics
            preprocessed_image = preprocess_input(rgb_image.astype("float32"))
            
            # Convert image to 4 Dimension (batch_size, height, width, channels)
            image_for_prediction = preprocessed_image.reshape(1, 224, 224, 3)
            
            # Make Prediction
            Y_pred = model.predict(image_for_prediction, verbose=0)
            max_confidence = float(np.max(Y_pred))
            predicted_class_idx = np.argmax(Y_pred)
            result = CLASS_NAMES[predicted_class_idx]
            st.session_state.prediction_result = result
            st.session_state.confidence = max_confidence
            
            # STEP 1: VALIDATION - Since all classes are tomato diseases, we validate based on confidence
            # Accept if confidence is above a threshold (e.g., 30%) OR if it's clearly a tomato class
            confidence_threshold = 0.30  # 30% confidence threshold
            
            if max_confidence >= confidence_threshold or result.startswith('Tomato-'):
                st.session_state.is_tomato = True
                st.session_state.step = 1
                st.success(f"✅ **Thank you for the clear image of your tomato leaf.** (Confidence: {max_confidence*100:.1f}%)")
                st.session_state.diagnosis = result
            else:
                # Low confidence - might not be a tomato leaf
                st.session_state.is_tomato = False
                st.session_state.step = 0
                st.warning(f"⚠️ **Low confidence prediction ({max_confidence*100:.1f}%).**")
                st.error("❌ **This does not appear to be a tomato leaf.** To provide an accurate diagnosis, please upload a clear, well-lit photo of the affected tomato leaf.")
                st.info("💡 **Tip:** Make sure the image shows a clear view of the tomato leaf, is well-lit, and focused.")
    else:
        st.warning("Please upload an image first.")

# STEP 2: DISEASE DIAGNOSIS (if tomato leaf validated)
if st.session_state.step >= 1 and st.session_state.is_tomato:
    st.markdown("---")
    st.header("Step 2: Disease Diagnosis")
    
    disease_name = st.session_state.diagnosis.split('-')[1].replace('_', ' ').title()
    
    # Special handling for healthy plants
    if 'healthy' in disease_name.lower():
        st.success(f"**Diagnosis:** Your tomato leaf appears **Healthy**! 🎉")
        st.info("No disease detected. Your plant is in good condition!")
    else:
        st.info(f"**Diagnosis:** This leaf shows classic signs of **{disease_name}**.")
    
    # Display confidence
    st.caption(f"Prediction Confidence: {st.session_state.confidence*100:.1f}%")
    
    # Only set step to 2 if we're not already past step 2 (to prevent resetting after form submission)
    if st.session_state.step < 2:
        st.session_state.step = 2

# STEP 3: CONTEXTUAL INQUIRY (after diagnosis)
if st.session_state.step == 2 and st.session_state.is_tomato:
    st.markdown("---")
    st.header("Step 3: Contextual Inquiry")
    st.markdown("To provide you with the most effective treatment plan, I need to understand your growing conditions. Please answer the following questions:")
    
    # Use form to prevent multiple submissions and infinite loops
    with st.form("contextual_questions_form", clear_on_submit=False):
        st.subheader("Growing Conditions Questionnaire")
        
        climate = st.selectbox(
            "**1. Is this plant growing in wet, humid conditions (e.g., frequent rain, poor drainage) or a dry, arid climate?**",
            ["Select an option...", "Wet/Humid conditions (frequent rain, poor drainage)", "Dry/Arid climate", "Moderate/Mixed conditions", "Indoor/Greenhouse"]
        )
        
        watering = st.selectbox(
            "**2. How do you typically water your plants?**",
            ["Select an option...", "Overhead sprinklers", "Watering at the base of plants", "Drip irrigation", "Hand watering (mixed methods)", "Other method"]
        )
        
        spread = st.selectbox(
            "**3. Have you noticed this issue on other nearby plants or just this one?**",
            ["Select an option...", "Just this one plant", "Multiple nearby plants", "Most/all plants in the area", "Not sure yet"]
        )
        
        submit_button = st.form_submit_button("📋 Submit Answers & Get Action Plan", use_container_width=True)
        
        # Handle form submission - ONLY executes when button is actually clicked
        # Streamlit automatically reruns after form submission, so we don't need st.rerun()
        if submit_button:
            # Validate all answers are selected
            if climate != "Select an option..." and watering != "Select an option..." and spread != "Select an option...":
                # Store answers in session state
                st.session_state.contextual_answers = {
                    'climate': climate,
                    'watering': watering,
                    'spread': spread
                }
                # Update step to 3 - Streamlit will automatically rerun after form submission
                # On the next rerun, step will be 3, so this form won't show and Step 4 will display
                st.session_state.step = 3
            else:
                # Show error but don't stop execution - let user fix and resubmit
                st.error("⚠️ Please answer all questions before submitting.")

# STEP 4: POWERFUL ACTION PLAN (after answers submitted)
if st.session_state.step >= 3 and st.session_state.is_tomato:
    # Check if we have all required data
    if st.session_state.contextual_answers and st.session_state.diagnosis:
        st.markdown("---")
        st.header("Step 4: Comprehensive Action Plan")
        st.markdown("Based on your diagnosis and growing conditions, here is your tailored action plan:")
        
        # Generate action plan
        try:
            action_plan = generate_action_plan(st.session_state.diagnosis, st.session_state.contextual_answers)
        except Exception as e:
            st.error(f"Error generating action plan: {e}")
            st.stop()
        
        # Display Immediate Actions
        st.subheader("🚨 Immediate Actions")
        for action in action_plan["immediate_actions"]:
            st.markdown(f"- {action}")
        
        # Display Organic & Cultural Remedies
        st.subheader("🌿 Organic & Cultural Remedies")
        for remedy in action_plan["organic_remedies"]:
            st.markdown(f"- {remedy}")
        
        # Display Chemical Treatment
        st.subheader("🧪 Chemical Treatment (If Necessary)")
        st.markdown(action_plan["chemical_treatment"])
        
        # Display Long-Term Prevention
        st.subheader("🌱 Long-Term Prevention")
        for prevention in action_plan["long_term_prevention"]:
            st.markdown(f"- {prevention}")
        
        st.markdown("---")
        st.markdown("### 📝 Additional Notes")
        st.info("Remember: Early intervention is key! Follow the immediate actions first, then implement the organic remedies. Only use chemical treatments if the situation is severe or spreading rapidly. Good luck with your tomato plants! 🌱")
        
        # Reset button
        if st.button("🔄 Start New Diagnosis"):
            for key in list(st.session_state.keys()):
                # Keep model_loaded to avoid re-showing the success message
                if key != 'model_loaded':
                    del st.session_state[key]
            st.rerun()
    else:
        # If we don't have the data, show error and allow retry
        st.warning("⚠️ Missing information. Please go back and complete the questionnaire.")
        if st.button("🔙 Back to Questions"):
            st.session_state.step = 2
            st.rerun()

# Footer
st.markdown("---")
st.caption("Agri-Bot | Expert AI Agronomist | Specializing in Tomato Cultivation")
st.caption(f"Model supports {len(CLASS_NAMES)} tomato disease classes")