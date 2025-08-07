import streamlit as st
st.set_page_config(page_title="AI Recipe Recommender", layout="wide")

# --- OTHER IMPORTS ---
import os
import re
import yaml
import numpy as np
import pandas as pd
from PIL import Image
from ast import literal_eval
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO

# --- PATH CONFIG ---
ROOT_DIR = os.getcwd()
DATA_YAML_PATH = os.path.join(ROOT_DIR, 'datasets', 'yolo_single_ingredient_dataset', 'data.yaml')
MODEL_WEIGHTS_PATH = os.path.join('models', 'yolo11s_100_epochs', 'weights', 'best.pt')
RECIPE_CSV = os.path.join(ROOT_DIR, 'food_ingredients_and_recipes_dataset_with_images', 'archive', 'recipes.csv')
# Download the images from the dataset in this link and place it in the specified folder so you can have the corresponding images for the recipes.
IMAGE_FOLDER = os.path.join(ROOT_DIR, 'food_ingredients_and_recipes_dataset_with_images', 'archive', 'images')
DEFAULT_IMAGE_PATH = os.path.join(ROOT_DIR, 'default_picture.png')

# --- LOAD MODELS AND DATA ---
@st.cache_resource
def load_models():
    yolo = YOLO(MODEL_WEIGHTS_PATH)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return yolo, embed_model

@st.cache_data
def load_data():
    with open(DATA_YAML_PATH, 'r') as file:
        class_names = yaml.safe_load(file)['names']

    df = pd.read_csv(RECIPE_CSV)
    df['Cleaned_Ingredients'] = df['Cleaned_Ingredients'].apply(literal_eval)
    lemmatizer = WordNetLemmatizer()

    df['Tokenized_Ingredients'] = df['Cleaned_Ingredients'].apply(
       lambda x: [word for ingredient in x for word in ingredient.split()]
    )

    df['Processed_Ingredients'] = df['Tokenized_Ingredients'].apply(
        lambda x: [lemmatizer.lemmatize(re.sub(r'[^a-zA-Z]', ' ', i.lower()).strip()) for i in x]
    )

    df['Processed_Ingredients'] = df['Processed_Ingredients'].apply(
        lambda x: list(filter(None, set(x)))
    )

    df['Ingredients_String'] = df['Processed_Ingredients'].apply(lambda x: " ".join(x))
    return df, class_names, lemmatizer

yolo_model, embed_model = load_models()
df, class_names, lemmatizer = load_data()
recipe_embeddings = embed_model.encode(df['Ingredients_String'].tolist())

SIMILARITY_THRESHOLD = 0.5

def filter_recommendations(similarities, df_subset):
    top_indices = np.argsort(similarities[0])[::-1]
    filtered = [(idx, similarities[0][idx]) for idx in top_indices if similarities[0][idx] >= SIMILARITY_THRESHOLD]
    return filtered[:10]  # top 10

def apply_filtering_mode(df, processed_ingredients, filter_mode):
    detected_set = set(processed_ingredients)

    if filter_mode == "Only the detected ingredients":
        return df[df['Processed_Ingredients'].apply(lambda x: set(x).issubset(detected_set))]
    elif filter_mode == "All the detected ingredients (but can contain others)":
        return df[df['Processed_Ingredients'].apply(lambda x: detected_set.issubset(set(x)))]
    else:  # "At least one of the detected ingredients"
        return df[df['Processed_Ingredients'].apply(lambda x: bool(set(x) & detected_set))]

def recommend_from_text(ingredient_text, filter_mode):
    processed_ingredients = [
        lemmatizer.lemmatize(re.sub(r'[^a-zA-Z]', ' ', ing.strip().lower()))
        for ing in ingredient_text.split(",") if ing.strip()
    ]
    processed_ingredients = list(set(processed_ingredients))

    filtered_df = apply_filtering_mode(df, processed_ingredients, filter_mode)
    if filtered_df.empty:
        return processed_ingredients, []

    query_embedding = embed_model.encode([" ".join(processed_ingredients)])
    recipe_embeddings_filtered = embed_model.encode(filtered_df['Ingredients_String'].tolist())
    similarities = cosine_similarity(query_embedding, recipe_embeddings_filtered)

    filtered_indices = filter_recommendations(similarities, filtered_df)

    results_list = []
    for idx, score in filtered_indices:
        row = filtered_df.iloc[idx]
        title = row['Title'] if pd.notna(row['Title']) else "Untitled Recipe"
        ingredients = row['Ingredients'] if pd.notna(row['Ingredients']) else "[]"
        instructions = row['Instructions'] if pd.notna(row['Instructions']) else "Not available"
        image_name = row['Image_Name']

        if not os.path.splitext(image_name)[1]:
            for ext in ['.jpg', '.jpeg', '.png']:
                if os.path.exists(os.path.join(IMAGE_FOLDER, image_name + ext)):
                    image_name += ext
                    break

        img_path = os.path.join(IMAGE_FOLDER, image_name)
        image = Image.open(img_path).convert("RGB") if os.path.exists(img_path) else Image.open(DEFAULT_IMAGE_PATH).convert("RGB")

        results_list.append({
            "title": title,
            "similarity": score,
            "ingredients": ingredients,
            "instructions": instructions,
            "image": image
        })

    return processed_ingredients, results_list

def recommend_from_image(image_path, filter_mode):
    results = yolo_model.predict(source=image_path, save=True, project="predictions", name="result", exist_ok=True)
    predicted_img_path = os.path.join("predictions", "result", os.path.basename(image_path))

    detected_classes = set()
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = class_names[class_id]
            detected_classes.add(class_name)

    if not detected_classes:
        return [], [], predicted_img_path

    processed_ingredients = list(set([
        lemmatizer.lemmatize(re.sub(r'[^a-zA-Z]', ' ', ing.lower()).strip())
        for ing in detected_classes
    ]))

    filtered_df = apply_filtering_mode(df, processed_ingredients, filter_mode)
    if filtered_df.empty:
        return processed_ingredients, [], predicted_img_path

    query_embedding = embed_model.encode([" ".join(processed_ingredients)])
    recipe_embeddings_filtered = embed_model.encode(filtered_df['Ingredients_String'].tolist())
    similarities = cosine_similarity(query_embedding, recipe_embeddings_filtered)

    filtered_indices = filter_recommendations(similarities, filtered_df)

    results_list = []
    for idx, score in filtered_indices:
        row = filtered_df.iloc[idx]
        title = row['Title'] if pd.notna(row['Title']) else "Untitled Recipe"
        ingredients = row['Ingredients'] if pd.notna(row['Ingredients']) else "[]"
        instructions = row['Instructions'] if pd.notna(row['Instructions']) else "Not available"
        image_name = row['Image_Name']

        if not os.path.splitext(image_name)[1]:
            for ext in ['.jpg', '.jpeg', '.png']:
                if os.path.exists(os.path.join(IMAGE_FOLDER, image_name + ext)):
                    image_name += ext
                    break

        img_path = os.path.join(IMAGE_FOLDER, image_name)
        image = Image.open(img_path).convert("RGB") if os.path.exists(img_path) else Image.open(DEFAULT_IMAGE_PATH).convert("RGB")

        results_list.append({
            "title": title,
            "similarity": score,
            "ingredients": ingredients,
            "instructions": instructions,
            "image": image
        })

    return processed_ingredients, results_list, predicted_img_path

# --- STREAMLIT UI ---
st.title("🥗 Intelligent Recipe Recommender")
st.markdown("### 🦾 Detectable Ingredients by the AI Model")
with st.expander("🍅 Click to view full list"):
    st.markdown(
        ", ".join(class_names),
        help="These are the ingredients our model can detect from images or text."
    )
st.write("Choose an input method to discover recipes from available ingredients.")

input_method = st.radio("Select Input Method:", ["Upload Image", "Type Ingredients"])

filter_mode = st.radio(
    "Recipe should contain:",
    [
        "At least one of the detected ingredients",
        "Only the detected ingredients",
        "All the detected ingredients (but can contain others)"
    ]
)

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("📸 Upload an ingredient image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img_path = os.path.join("temp_input.jpg")
        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())

        original_image = Image.open(img_path).convert("RGB")

        with st.spinner("🔍 Detecting ingredients..."):
            ingredients, recommendations, predicted_img_path = recommend_from_image(img_path, filter_mode)

        st.subheader("📷 Ingredient Detection")
        col1, col2 = st.columns(2)
        col1.image(original_image, caption="Original Image", width=350)
        if os.path.exists(predicted_img_path):
            pred_img = Image.open(predicted_img_path).convert("RGB")
            col2.image(pred_img, caption="Detected Ingredients", width=350)

        if not ingredients:
            st.warning("⚠️ No ingredients detected.")
        else:
            st.success(f"✅ Detected Ingredients: {', '.join(ingredients)}")
            st.subheader("🍽 Recommended Recipes")
            if not recommendations:
                st.info("🔍 No recipes found that match the selected filter and similarity threshold (≥ 0.5). Try changing the filter or using more common ingredients.")
            for rec in recommendations:
                st.markdown(f"### {rec['title']} (Similarity: {rec['similarity']:.2f})")
                cols = st.columns([1, 2])
                cols[0].image(rec["image"], caption=rec['title'])
                cols[1].markdown(f"**Ingredients:** {rec['ingredients']}\n\n**Instructions:** {rec['instructions']}")

elif input_method == "Type Ingredients":
    typed_input = st.text_area("📝 Enter comma-separated ingredients (e.g., tomato, onion, garlic)")
    if st.button("Recommend Recipes"):
        with st.spinner("🔍 Finding recipes..."):
            ingredients, recommendations = recommend_from_text(typed_input, filter_mode)

        if not ingredients:
            st.warning("⚠️ No valid ingredients entered.")
        else:
            st.success(f"✅ You entered: {', '.join(ingredients)}")
            st.subheader("🍽 Recommended Recipes")
            if not recommendations:
                st.info("🔍 No recipes found that match the selected filter and similarity threshold (≥ 0.5). Try changing the filter or using more common ingredients.")
            for rec in recommendations:
                st.markdown(f"### {rec['title']} (Similarity: {rec['similarity']:.2f})")
                cols = st.columns([1, 2])
                cols[0].image(rec["image"], caption=rec['title'])
                cols[1].markdown(f"**Ingredients:** {rec['ingredients']}\n\n**Instructions:** {rec['instructions']}")