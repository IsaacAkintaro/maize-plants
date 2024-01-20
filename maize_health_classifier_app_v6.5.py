import streamlit as st
from roboflow import Roboflow
from PIL import Image
from io import BytesIO

# Function to get predictions from Roboflow model
def get_prediction(image_bytes):
    ## Create a .streamlit/secrets.toml with the entry, replacing YourKey with the key from Roboflow: api_key="YourKey"
    ## Don't commit secrets.toml. On Sharing, add the same line to ☰ -> Settings -> Secrets
    access_token = st.secrets["api_key"]
    project_name = "maize-catergories"
    version = "1" # Replace with your version number
    
    rf = Roboflow(api_key=access_token)
    project = rf.workspace().project(project_name)
    model = project.version(version).model

    # Save the uploaded image to a temporary file for prediction
    image = Image.open(BytesIO(image_bytes))
    # convert from RGBA if needed i.e. png
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image.save("temp_image.jpg")
    print("Saved image for prediction.")
    prediction = model.predict("temp_image.jpg").json()
    print("Received prediction.")
    return prediction

# Streamlit app interface
st.set_page_config(page_title="Maize Plant Health Check")

st.write("## Check the health of your Maize Plant")
st.write(
    ":herb::corn: Try uploading an image to discover the health of your maize plant in its [early stages](https://www.masseeds.com/nos-dossiers/key-growth-stages-maize) of growth with this app, leveraging a detailed dataset for accurate disease diagnosis and analysis."
    " Special thanks to [The KaraAgro AI Maize Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CXUMDS) which was used to train the machine learning model. :smile:"
)


# Load our images
image1 = Image.open("./images/healthy_maize.PNG")
image2 = Image.open("./images/Maize_Streak_Disease.PNG")
image3 = Image.open("./images/fall_armyworm.PNG")

# Set the desired fixed height for all images
fixed_image_height = 300

# # Streamlit app interface

col1, col2, col3 = st.columns(3)
col1.write("Healthy Maize Plant")
col1.image(image1, use_column_width=True)
col2.write("Maize Streak Disease")
col2.image(image2, use_column_width=True)
col3.write("Fall Armyworm Damage")
col3.image(image3, use_column_width=True)


# Initialize prediction as None
prediction = None

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    # Get predictions
    with st.spinner('Analyzing the image...'):
        prediction = get_prediction(uploaded_file.getvalue())

    st.success('Analysis complete.')

    # Display predictions individually
    st.write(f"Prediction Time:", prediction['predictions'][0]['time'])

    st.write("Top Prediction:", prediction['predictions'][0]['top'], "with confidence", prediction['predictions'][0]['confidence'])

