import streamlit as st
from PIL import Image
from pathlib import Path
from streamlit_image_comparison import image_comparison
from image_processes import *
from feature_extraction import harris_response, compute_gradients, detect_harris_keypoints, compute_descriptors, draw_keypoints
from wordcloud import WordCloud
import pandas as pd
from io import BytesIO
from db import *

init_db()


# Sayfa basligi
st.set_page_config(
    page_title="Image Editor App", 
    layout="wide",
    page_icon="logo.png")


current_file_path = Path(__file__)
parent_dir = current_file_path.parent
main_dir = parent_dir.parent

# Görseli kesme işlevi
def display_comparison(image_before, image_after, slider_value):
    width = int(image_before.width * (slider_value / 100))
    left = image_before.crop((0, 0, width, image_before.height))  # Önceki görüntüyü kes
    right = image_after.crop((width, 0, image_after.width, image_after.height))  # Sonraki görüntüyü kes
    
    # Görselleri streamlit ekranında göster
    st.image([left, right], width=500)

# Custom CSS for full-width buttons, hiding borders, hover effect, and selected effect
st.markdown("""
    <style>
    .stButton button {
        width: 100%;  /* Adjust the width as needed */
        border: none;
        margin: 5px 0;  /* Add some margin for better spacing */
        background-color: transparent;  /* Remove background color */
         /* Inherit text color */
    }
    .stButton button:hover {
        background-color: #e6e6e6;  /* Hover color */
         /* Text color on hover */
        color: #000000;  /* Text color on hover */
    }
    .stButton button.selected {
        background-color: #e6e6e6;  /* Selected color */
         /* Text color on selected */
        color: #000000;
    }
    .stButton applybutton {
        width: 100%;  /* Full width */
        background-color: #95a8bd;  /* Blue background */
        color: white;  /* White text */
        border: none;  /* Remove border */
        padding: 10px;  /* Add padding */
        font-size: 16px;  /* Font size */
        cursor: pointer;  /* Pointer cursor on hover */
        border-radius: 5px;  /* Rounded corners */
        color: #000000;  /* White text color */
    }
    .stButton applybutton:hover {
        background-color: #c4dbf5 ;  /* Darker blue on hover */
    }
            
    .full-height {
        height: 100vh;  /* Full viewport height */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .content {
        flex: 1;
        overflow: auto;
    }
    .header {
        background-color: transparent;  /* Darker background color */
        margin-top: -60px;  /* Reduce padding to make the header smaller */
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        z-index: 1000;
        border-bottom: 2px solid #dd140e;  /* Add bottom border */
    }
    .header h2 {
        cursor: pointer;
    }
    .user-info {
        display: flex;
        
        align-items: center;
    }
    .user-info span {
        font-size: 20px;
    }     
    .user-info img {
        border-radius: 50%;
        width: 50px;  /* Adjust the width as needed */
        height: 50px;  /* Adjust the height as needed */
        margin-left: 20px;
    }
    .sidebar .sidebar-content h132 {
        text-align: center;  /* Center align the sidebar title */
    }
    </style>
    """, unsafe_allow_html=True)



# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Dashboard'  # Default page

# Function to create a button with selected state
def create_button(label):

    if st.session_state.page == label:
        button_css = "selected"

    else:
        button_css = ""

    return st.markdown(f"""
        <div class="stButton">
            <button class="{button_css}" onclick="window.location.href='?page={label}'">{label}</button>
        </div>
    """, unsafe_allow_html=True)

# Header
st.markdown(
    """
    <div class="header" style='background-color: transparent; padding: 10px; display: flex; justify-content: space-between; align-items: center;'>
    <div style='display: flex; align-items: center;'>
            <img src='https://upload.wikimedia.org/wikipedia/commons/c/cb/Processing_2021_logo.svg' style='width: 50px; height: 50px; margin-left: 10px;' />
        <div style='display: flex; align-items: center;'>
            <h2 style='margin: 0; margin-left: 15px;'>
                Image Editor App
            </h2>
        </div>
    </div>
    """, unsafe_allow_html=True
)

# Sidebar (Navigasyon Menusu)
st.sidebar.markdown("<h1 style='text-align: center; margin-top: -20px; margin-bottom: 20px;'>Menü</h1>", unsafe_allow_html=True)

if st.sidebar.button('Dashboard'):
    st.session_state.page = 'Dashboard'

if st.sidebar.button('Color & Contrast Processing'):
    st.session_state.page = 'Contrast'

if st.sidebar.button('Blurring & Sharpening'):
    st.session_state.page = 'Blurring'

if st.sidebar.button('Edge Detection'):
    st.session_state.page = 'Edge'

if st.sidebar.button('Feature Detection'):
    st.session_state.page = 'Feature'

if st.sidebar.button('Thresholding & Slicing'):
    st.session_state.page = 'Thresholding'

if st.sidebar.button("Database"):
    st.session_state.page = "Database"

if st.sidebar.button('About'):
    st.session_state.page = 'About'

# Layout
col2, col3 = st.columns([6, 2.3])


# Orta Ana Icerik
with col2:


    ############################################## DASHBOARD #######################################################

    if st.session_state.page == 'Dashboard':
        st.markdown("<h3 style='text-align: center; margin-top : 50px'>Dashboard</h3>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size: 18px;'><br>Welcome to Image Editor App!<br><br>This is an image editor app that allows you to edit images using various tools.</p>",
            unsafe_allow_html=True
        )

        total, unique, per_technique = get_statistics()

        # Display two cards side by side
        st.markdown(f"""
            <div style='display: flex; justify-content: flex-start; gap: 20px; margin-top: 40px; margin-left: 0px; width: 120%;'>
            <div style='display: flex; align-items: center; justify-content: center; background-color: #f0f0f0; width: 45vh; height: 25vh; border-radius: 10px; padding: 20px;'>
            <div style='width: 80px; height: 8vh; border: 2px solid transparent; border-radius: 50%; display: flex; justify-content: center; align-items: center; background-color: #f0f0f0; margin-right: 50px; margin-left: 10px;'>
            <img src='https://cdn-icons-png.freepik.com/128/1159/1159633.png' alt='Icon' style='width: 60px; height: 60px;' />
            </div>
            <div style='text-align: left;'>
            <h4 style='margin-top: 15px; color: #000000'>Total:</h4>
            <h3 style='margin-top: 5px; color: #dd140e;'>{total} image(s)</h3>
            <p style='margin-bottom: 40px; color: #000000'>have been edited in this project.</p>
            </div>
            </div>
            <div style='display: flex; align-items: center; justify-content: center; background-color: #f0f0f0; width: 45vh; height: 25vh; border-radius: 10px; padding: 20px;'>
            <div style='width: 80px; height: 8vh; border: 2px solid transparent; border-radius: 50%; display: flex; justify-content: center; align-items: center; background-color: #f0f0f0; margin-right: 50px; margin-left: 10px;'>
            <img src='https://cdn-icons-png.freepik.com/128/11600/11600124.png' alt='Icon' style='width: 60px; height: 60px;' />
            </div>
            <div style='text-align: left;'>
            <h4 style='margin-top: 15px; color: #000000'>Total:</h4>
            <h3 style='margin-top: 5px; color: #dd140e;'>4 main categories</h3>
            <p style='margin-bottom: 40px; color: #000000'>have been used in that project.</p>
            </div>
            </div>
            <div style='display: flex; align-items: center; justify-content: center; background-color: #f0f0f0; width: 45vh; height: 25vh; border-radius: 10px; padding: 20px;'>
            <div style='width: 80px; height: 8vh; border: 2px solid transparent; border-radius: 50%; display: flex; justify-content: center; align-items: center; background-color: #f0f0f0; margin-right: 50px; margin-left: 10px;'>
            <img src='https://cdn-icons-png.freepik.com/128/9752/9752076.png' alt='Icon' style='width: 60px; height: 60px;' />
            </div>
            <div style='text-align: left;'>
            <h4 style='margin-top: 15px; color: #000000'>Total:</h4>
            <h3 style='margin-top: 5px; color: #dd140e;'>17 techniques</h3>
            <p style='margin-bottom: 40px; color: #000000'>have been implemented to that project.</p>
            </div>
            </div>
            </div>
        """, unsafe_allow_html=True)


        # Sabit kelime listesi
        kelimeler = {
            "Python": 10,
            "OpenCV": 8,
            "Streamlit": 7,
            "Histogram": 5,
            "Filter": 6,
            "ImageProcessing": 10,
            "Sobel": 4,
            "Gaussian": 6,
            "Canny": 7,
            "Threshold": 5,
            "Blurring": 6,
            "Sharpening": 5,
            "EdgeDetection": 8,
            "FeatureDetection": 7}

        # Alternatif: farklı sıklıklar
        # word_freq = {"Python": 10, "OpenCV": 8, "Streamlit": 7, "Histogram": 5, "Veri": 3}

        # Kelime bulutu oluştur
        wordcloud = WordCloud(width=800,
                            height=400,
                            background_color="white",
                            colormap="Set2").generate_from_frequencies(kelimeler)

        # Görselleştir
        # Kelime bulutu oluştur
        wordcloud = WordCloud(width=800, height=400, mode="RGBA", background_color=None).generate_from_frequencies(kelimeler)

        # Görseli elde et
        image = wordcloud.to_image()

        # Streamlit ile göster
        st.markdown("<div style='margin-left: 60px;'><br><br><br><br></div>", unsafe_allow_html=True)  # Add left margin and top margin
        st.image(image, caption="Word Cloud", use_container_width=True)


    ############################# COLOR & CONTRAST PROCESSING #############################

    if st.session_state.page == 'Contrast':
        st.markdown("<h3 style='text-align: center; margin-top : 50px'>Color & Contrast Processing</h3>", unsafe_allow_html=True)

        st.markdown(
            "<p style='font-size: 18px;'><br>Select a tool for color and contrast processing.</p>",
            unsafe_allow_html=True
        )
        
        # Add a combobox for selecting a quick tool
        tool = st.selectbox(
            "Choose a tool:",
            ["Saturation", "Histogram Equalization", "Gamma Correction"]
        )
        
        
        # Display the selected tool
        st.markdown(f"<h3 style='margin-top: 3vh;'>{tool}</h3>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 18px;'>Upload an image to apply the selected tool.</p>", unsafe_allow_html=True)
        
        placeholder = st.empty()

        if tool == "Saturation":
            # File uploader for image uploading
            
            uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

            # Görsel yüklendiyse, altta gösterelim
            if uploaded_file is not None:
                # Görseli açalım
                image = Image.open(uploaded_file)
                
                placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1000)
                # Görseli Streamlit'te gösterelim
        

                # Add a button to apply the selected tool
                if st.button("Apply Tool", key="apply_button"):

                    # Check if an image is uploaded
                    if uploaded_file is None:
                        st.warning("Please upload an image before applying the tool.")
                        
                    else:
                        file_type = uploaded_file.type.split('/')[1]  # Get the file type (e.g., 'jpeg', 'png')
                        file_name = f"uploaded_image.{file_type}"  # Create a file name for the uploaded image

                        # Filtre fonksiyonu cagrilacak
                        # Apply filter adjustment

                        placeholder.empty()  # Clear the placeholder

                        # Enhance the saturation of the image
                        enhanced_image = Saturation(image)

                        # Display the enhanced image
                        # st.image(enhanced_image, caption="Saturation Adjusted Image", use_container_width=True)
                        st.success(f"{tool} has been applied successfully. Check both of before-after images by slider!")
                        image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1000)
                            
                        # ✅ Log to DB
                        # Log the process to the database
                        log_process(file_name, tool)

                        
                        # Dosyayı indirilebilir hale getirme
                        # Input field for custom file name
                        custom_file_name = st.text_input("Enter file name (without extension):", value="edited_image")

                        # Update the file name with the custom input
                        file_name = f"{custom_file_name}.{file_type}"

                        # Save the enhanced image to a BytesIO object
                        buffer = BytesIO()
                        enhanced_image.save(buffer, format=file_type.upper())
                        buffer.seek(0)

                        # Create a download button
                        st.download_button(
                            label="Download Edited Image",
                            data=buffer,
                            file_name=file_name,
                            mime=f"image/{file_type}",
                            key="download_button",
                            help="Click to download the edited image",
                            use_container_width=True  # Adjust the button width to fit the container
                        )

                        
                        
        elif tool == "Histogram Equalization":
            # File uploader for image uploading
            
            uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

            # Görsel yüklendiyse, altta gösterelim
            if uploaded_file is not None:
                # Görseli açalım
                image = Image.open(uploaded_file)
                
                placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1000)
                # Görseli Streamlit'te gösterelim
        

                # Add a button to apply the selected tool
                if st.button("Apply Tool", key="apply_button"):

                    # Check if an image is uploaded
                    if uploaded_file is None:
                        st.warning("Please upload an image before applying the tool.")
                        
                    else:
                        file_type = uploaded_file.type.split('/')[1]  # Get the file type (e.g., 'jpeg', 'png')
                        file_name = f"uploaded_image.{file_type}"  # Create a file name for the uploaded image

                        # Filtre fonksiyonu cagrilacak
                        # Apply filter adjustment

                        placeholder.empty()  # Clear the placeholder

                        enhanced_image = HistogramEqualization(image)

                        st.success(f"{tool} has been applied successfully. Check both of before-after images by slider!")
                        image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1000)
                            
                        # ✅ Log to DB
                        # Log the process to the database
                        log_process(file_name, tool)

                        
                        # Dosyayı indirilebilir hale getirme
                        # Input field for custom file name
                        custom_file_name = st.text_input("Enter file name (without extension):", value="edited_image")

                        # Update the file name with the custom input
                        file_name = f"{custom_file_name}.{file_type}"

                        # Save the enhanced image to a BytesIO object
                        buffer = BytesIO()
                        enhanced_image.save(buffer, format=file_type.upper())
                        buffer.seek(0)

                        # Create a download button
                        st.download_button(
                            label="Download Edited Image",
                            data=buffer,
                            file_name=file_name,
                            mime=f"image/{file_type}",
                            key="download_button",
                            help="Click to download the edited image",
                            use_container_width=True  # Adjust the button width to fit the container
                        )



        elif tool == "Gamma Correction":

            # File uploader for image uploading
            
            uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

            # Görsel yüklendiyse, altta gösterelim
            if uploaded_file is not None:
                # Görseli açalım
                image = Image.open(uploaded_file)
                
                placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1000)
                # Görseli Streamlit'te gösterelim
                # Add a slider for gamma value
                gamma_value = st.slider("Select Gamma Value", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

                # Add a button to apply the selected tool
                if st.button("Apply Tool", key="apply_button"):

                    # Check if an image is uploaded
                    if uploaded_file is None:
                        st.warning("Please upload an image before applying the tool.")
                        
                    else:
                        file_type = uploaded_file.type.split('/')[1]  # Get the file type (e.g., 'jpeg', 'png')
                        file_name = f"uploaded_image.{file_type}"  # Create a file name for the uploaded image

                        # Filtre fonksiyonu cagrilacak
                        # Apply filter adjustment

                        placeholder.empty()  # Clear the placeholder

                        enhanced_image = GammaCorrection(image, gamma_value)

                        st.success(f"{tool} has been applied successfully. Check both of before-after images by slider!")
                        image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1000)
                            
                        # ✅ Log to DB
                        # Log the process to the database
                        log_process(file_name, tool)

                        
                        # Dosyayı indirilebilir hale getirme
                        # Input field for custom file name
                        custom_file_name = st.text_input("Enter file name (without extension):", value="edited_image")

                        # Update the file name with the custom input
                        file_name = f"{custom_file_name}.{file_type}"

                        # Save the enhanced image to a BytesIO object
                        buffer = BytesIO()
                        enhanced_image.save(buffer, format=file_type.upper())
                        buffer.seek(0)

                        # Create a download button
                        st.download_button(
                            label="Download Edited Image",
                            data=buffer,
                            file_name=file_name,
                            mime=f"image/{file_type}",
                            key="download_button",
                            help="Click to download the edited image",
                            use_container_width=True  # Adjust the button width to fit the container
                        )


    ############################### BLURRING & SHARPENING #############################

    if st.session_state.page == 'Blurring':
        st.markdown("<h3 style='text-align: center; margin-top : 50px'>Blurring & Sharpening</h3>", unsafe_allow_html=True)

        st.markdown(
            "<p style='font-size: 18px;'><br>Select a tool for blurring and sharpening.</p>",
            unsafe_allow_html=True
        )
        
        # Add a combobox for selecting a quick tool
        tool = st.selectbox(
            "Choose a tool:",
            ["Gaussian Blurring", "Median Blurring", "Average Blurring", "Bilateral Filtering"]
        )

        # Display the selected tool
        st.markdown(f"<h3 style='margin-top: 3vh;'>{tool}</h3>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 18px;'>Upload an image to apply the selected tool.</p>", unsafe_allow_html=True)
        
        placeholder = st.empty()

        match tool:
            case "Gaussian Blurring":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1000)
                    # Görseli Streamlit'te gösterelim

                    # Add a input box for kernel dimensions
                    kernel_size = st.number_input("Kernel Size", min_value=3, max_value=15, value=3, step=2)
                    sigma = st.number_input("Sigma", min_value=0, max_value=5, value=0, step=1)


                     # Add a button to apply the selected tool
                    if st.button("Apply Tool", key="apply_button"):

                        # Check if an image is uploaded
                        if uploaded_file is None:
                            st.warning("Please upload an image before applying the tool.")
                            
                        else:
                            file_type = uploaded_file.type.split('/')[1]  # Get the file type (e.g., 'jpeg', 'png')
                            file_name = f"uploaded_image.{file_type}"  # Create a file name for the uploaded image

                            # Filtre fonksiyonu cagrilacak
                            # Apply filter adjustment

                            placeholder.empty()  # Clear the placeholder


                            enhanced_image = GaussianBlur(image, kernel_size, sigma)

                            st.success(f"{tool} has been applied successfully. Check both of before-after images by slider!")
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1000)
                            
                            # ✅ Log to DB
                            # Log the process to the database
                            log_process(file_name, tool)
                                                       

                            # Dosyayı indirilebilir hale getirme
                            # Input field for custom file name
                            custom_file_name = st.text_input("Enter file name (without extension):", value="edited_image")

                            # Update the file name with the custom input
                            file_name = f"{custom_file_name}.{file_type}"

                            # Save the enhanced image to a BytesIO object
                            buffer = BytesIO()
                            enhanced_image.save(buffer, format=file_type.upper())
                            buffer.seek(0)

                            # Create a download button
                            st.download_button(
                                label="Download Edited Image",
                                data=buffer,
                                file_name=file_name,
                                mime=f"image/{file_type}",
                                key="download_button",
                                help="Click to download the edited image",
                                use_container_width=True  # Adjust the button width to fit the container
                            )


            case "Median Blurring":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1000)
                    # Görseli Streamlit'te gösterelim

                    # Add a input box for kernel dimensions
                    kernel_size = st.number_input("Kernel Size", min_value=3, max_value=15, value=3, step=2)

                     # Add a button to apply the selected tool
                    if st.button("Apply Tool", key="apply_button"):

                        # Check if an image is uploaded
                        if uploaded_file is None:
                            st.warning("Please upload an image before applying the tool.")
                            
                        else:
                            file_type = uploaded_file.type.split('/')[1]  # Get the file type (e.g., 'jpeg', 'png')
                            file_name = f"uploaded_image.{file_type}"  # Create a file name for the uploaded image

                            # Filtre fonksiyonu cagrilacak
                            # Apply filter adjustment

                            placeholder.empty()  # Clear the placeholder


                            enhanced_image = MedianBlur(image, kernel_size)

                            st.success(f"{tool} has been applied successfully. Check both of before-after images by slider!")
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1000)
                            
                            # ✅ Log to DB
                            # Log the process to the database
                            log_process(file_name, tool)
                                                     
                            # Dosyayı indirilebilir hale getirme
                            # Input field for custom file name
                            custom_file_name = st.text_input("Enter file name (without extension):", value="edited_image")

                            # Update the file name with the custom input
                            file_name = f"{custom_file_name}.{file_type}"

                            # Save the enhanced image to a BytesIO object
                            buffer = BytesIO()
                            enhanced_image.save(buffer, format=file_type.upper())
                            buffer.seek(0)

                            # Create a download button
                            st.download_button(
                                label="Download Edited Image",
                                data=buffer,
                                file_name=file_name,
                                mime=f"image/{file_type}",
                                key="download_button",
                                help="Click to download the edited image",
                                use_container_width=True  # Adjust the button width to fit the container
                            )
        

            case "Average Blurring":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1000)
                    # Görseli Streamlit'te gösterelim

                    # Add a input box for kernel dimensions
                    kernel_size = st.number_input("Kernel Size", min_value=3, max_value=15, value=3, step=2)

                     # Add a button to apply the selected tool
                    if st.button("Apply Tool", key="apply_button"):

                        # Check if an image is uploaded
                        if uploaded_file is None:
                            st.warning("Please upload an image before applying the tool.")
                            
                        else:
                            file_type = uploaded_file.type.split('/')[1]  # Get the file type (e.g., 'jpeg', 'png')
                            file_name = f"uploaded_image.{file_type}"  # Create a file name for the uploaded image

                            # Filtre fonksiyonu cagrilacak
                            # Apply filter adjustment

                            placeholder.empty()  # Clear the placeholder


                            enhanced_image = AverageBlur(image, kernel_size)

                            st.success(f"{tool} has been applied successfully. Check both of before-after images by slider!")
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1000)
                            
                            # ✅ Log to DB
                            # Log the process to the database
                            log_process(file_name, tool)

                        
                            # Dosyayı indirilebilir hale getirme
                            # Input field for custom file name
                            custom_file_name = st.text_input("Enter file name (without extension):", value="edited_image")

                            # Update the file name with the custom input
                            file_name = f"{custom_file_name}.{file_type}"

                            # Save the enhanced image to a BytesIO object
                            buffer = BytesIO()
                            enhanced_image.save(buffer, format=file_type.upper())
                            buffer.seek(0)

                            # Create a download button
                            st.download_button(
                                label="Download Edited Image",
                                data=buffer,
                                file_name=file_name,
                                mime=f"image/{file_type}",
                                key="download_button",
                                help="Click to download the edited image",
                                use_container_width=True  # Adjust the button width to fit the container
                            )
                
            case "Bilateral Filtering":
                # File uploader for image uploading
                
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1000)
                    # Görseli Streamlit'te gösterelim
                    # Add a slider for filter values
                    
                    diameter = st.slider("Diameter", min_value=-1, max_value=100, value=-1, step=1)
                    sigmaColor = st.slider("Sigma Color", min_value=0, max_value=200, value=25, step=1)
                    sigmaSpace = st.slider("Sigma Space", min_value=0, max_value=200, value=25, step=1)

                    # Add a button to apply the selected tool
                    if st.button("Apply Tool", key="apply_button"):

                        # Check if an image is uploaded
                        if uploaded_file is None:
                            st.warning("Please upload an image before applying the tool.")
                            
                        else:
                            file_type = uploaded_file.type.split('/')[1]  # Get the file type (e.g., 'jpeg', 'png')
                            file_name = f"uploaded_image.{file_type}"  # Create a file name for the uploaded image

                            # Filtre fonksiyonu cagrilacak
                            # Apply filter adjustment

                            placeholder.empty()  # Clear the placeholder


                            enhanced_image = BilateralFiltering(image, int(diameter), int(sigmaColor), int(sigmaSpace))

                            st.success(f"{tool} has been applied successfully. Check both of before-after images by slider!")
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1000)
                            
                            # ✅ Log to DB
                            # Log the process to the database
                            log_process(file_name, tool)
                            
                            # Dosyayı indirilebilir hale getirme
                            # Input field for custom file name
                            custom_file_name = st.text_input("Enter file name (without extension):", value="edited_image")

                            # Update the file name with the custom input
                            file_name = f"{custom_file_name}.{file_type}"

                            # Save the enhanced image to a BytesIO object
                            buffer = BytesIO()
                            enhanced_image.save(buffer, format=file_type.upper())
                            buffer.seek(0)

                            # Create a download button
                            st.download_button(
                                label="Download Edited Image",
                                data=buffer,
                                file_name=file_name,
                                mime=f"image/{file_type}",
                                key="download_button",
                                help="Click to download the edited image",
                                use_container_width=True  # Adjust the button width to fit the container
                            )
                            
    ################################## EDGE DETECTION #############################

            
    if st.session_state.page == 'Edge':
        st.markdown("<h3 style='text-align: center; margin-top : 50px'>Edge & Feature Detection</h3>", unsafe_allow_html=True)

        st.markdown(
            "<p style='font-size: 18px;'><br>Select a tool for edge and feature detection.</p>",
        unsafe_allow_html=True
        )

        # Add a combobox for selecting a quick tool
        tool = st.selectbox(
            "Choose a quick tool:",
            ["SOBEL Operator", "Laplacian of Gaussian (LoG)", "Prewitt Operator", "Roberts Cross Operator", "Scharr Operator", "Canny Edge Detection"]
        )

        # Display the selected tool
        st.markdown(f"<h3 style='margin-top: 3vh;'>{tool}</h3>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 18px;'>Upload an image to apply the selected tool.</p>", unsafe_allow_html=True)
        
        placeholder = st.empty()

        match tool:
            case "SOBEL Operator":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
            
                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1000)
                    # Görseli Streamlit'te gösterelim
                    # Add a slider for filter values
                    
                    dx = st.slider("dx", min_value=0, max_value=1, value=1, step=1)
                    dy = st.slider("dy", min_value=0, max_value=1, value=1, step=1)
                    ksize = st.slider("ksize", min_value=3, max_value=15, value=3, step=2) # 3-5-7-9-11-13-15

                    # Add a button to apply the selected tool
                    if st.button("Apply Tool", key="apply_button"):

                        # Check if an image is uploaded
                        if uploaded_file is None:
                            st.warning("Please upload an image before applying the tool.")
                            
                        else:
                            file_type = uploaded_file.type.split('/')[1]  # Get the file type (e.g., 'jpeg', 'png')
                            file_name = f"uploaded_image.{file_type}"  # Create a file name for the uploaded image

                            # Filtre fonksiyonu cagrilacak
                            # Apply filter adjustment

                            placeholder.empty()  # Clear the placeholder

                            enhanced_image = SobelOperator(image, int(dx), int(dy), int(ksize))

                            st.success(f"{tool} has been applied successfully. Check both of before-after images by slider!")
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1000)
                            
                            # ✅ Log to DB
                            # Log the process to the database
                            log_process(file_name, tool)                     
                        
                            # Dosyayı indirilebilir hale getirme
                            # Input field for custom file name
                            custom_file_name = st.text_input("Enter file name (without extension):", value="edited_image")

                            # Update the file name with the custom input
                            file_name = f"{custom_file_name}.{file_type}"

                            # Save the enhanced image to a BytesIO object
                            buffer = BytesIO()
                            enhanced_image.save(buffer, format=file_type.upper())
                            buffer.seek(0)

                            # Create a download button
                            st.download_button(
                                label="Download Edited Image",
                                data=buffer,
                                file_name=file_name,
                                mime=f"image/{file_type}",
                                key="download_button",
                                help="Click to download the edited image",
                                use_container_width=True  # Adjust the button width to fit the container
                            )


            case "Laplacian of Gaussian (LoG)":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1000)
                    # Görseli Streamlit'te gösterelim
                    # Add a slider for filter values

                    ksize = st.slider("ksize", min_value=3, max_value=15, value=3, step=2) # 3-5-7-9-11-13-15
                    sigma = st.slider("sigma", min_value=0.0, max_value=5.0, value=1.0, step=0.1)


                    # Add a button to apply the selected tool
                    if st.button("Apply Tool", key="apply_button"):

                        # Check if an image is uploaded
                        if uploaded_file is None:
                            st.warning("Please upload an image before applying the tool.")
                            
                        else:
                            file_type = uploaded_file.type.split('/')[1]  # Get the file type (e.g., 'jpeg', 'png')
                            file_name = f"uploaded_image.{file_type}"  # Create a file name for the uploaded image

                            # Filtre fonksiyonu cagrilacak
                            # Apply filter adjustment

                            placeholder.empty()  # Clear the placeholder

                            enhanced_image = LaplacianOfGaussian(image, ksize, sigma)

                            st.success(f"{tool} has been applied successfully. Check both of before-after images by slider!")
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1000)
                            
                            # ✅ Log to DB
                            # Log the process to the database
                            log_process(file_name, tool)
                                                    
                            # Dosyayı indirilebilir hale getirme
                            # Input field for custom file name
                            custom_file_name = st.text_input("Enter file name (without extension):", value="edited_image")

                            # Update the file name with the custom input
                            file_name = f"{custom_file_name}.{file_type}"

                            # Save the enhanced image to a BytesIO object
                            buffer = BytesIO()
                            enhanced_image.save(buffer, format=file_type.upper())
                            buffer.seek(0)

                            # Create a download button
                            st.download_button(
                                label="Download Edited Image",
                                data=buffer,
                                file_name=file_name,
                                mime=f"image/{file_type}",
                                key="download_button",
                                help="Click to download the edited image",
                                use_container_width=True  # Adjust the button width to fit the container
                            )


            case "Prewitt Operator":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1000)
                    # Görseli Streamlit'te gösterelim
                    # Add a slider for filter values

                    dx = st.slider("dx", min_value=0, max_value=1, value=1, step=1)
                    dy = st.slider("dy", min_value=0, max_value=1, value=1, step=1)

                    # Add a button to apply the selected tool
                    if st.button("Apply Tool", key="apply_button"):

                        # Check if an image is uploaded
                        if uploaded_file is None:
                            st.warning("Please upload an image before applying the tool.")
                            
                        else:
                            file_type = uploaded_file.type.split('/')[1]  # Get the file type (e.g., 'jpeg', 'png')
                            file_name = f"uploaded_image.{file_type}"  # Create a file name for the uploaded image

                            # Filtre fonksiyonu cagrilacak
                            # Apply filter adjustment

                            placeholder.empty()  # Clear the placeholder

                            enhanced_image = PrewittOperator(image, int(dx), int(dy))

                            st.success(f"{tool} has been applied successfully. Check both of before-after images by slider!")
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1000)
                            
                            # ✅ Log to DB
                            # Log the process to the database
                            log_process(file_name, tool)                           
                            
                            # Dosyayı indirilebilir hale getirme
                            # Input field for custom file name
                            custom_file_name = st.text_input("Enter file name (without extension):", value="edited_image")

                            # Update the file name with the custom input
                            file_name = f"{custom_file_name}.{file_type}"

                            # Save the enhanced image to a BytesIO object
                            buffer = BytesIO()
                            enhanced_image.save(buffer, format=file_type.upper())
                            buffer.seek(0)

                            # Create a download button
                            st.download_button(
                                label="Download Edited Image",
                                data=buffer,
                                file_name=file_name,
                                mime=f"image/{file_type}",
                                key="download_button",
                                help="Click to download the edited image",
                                use_container_width=True  # Adjust the button width to fit the container
                            )


            case "Roberts Cross Operator":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1000)
                    # Görseli Streamlit'te gösterelim
                    # Add a slider for filter values

                    dx = st.slider("dx", min_value=0, max_value=1, value=1, step=1)
                    dy = st.slider("dy", min_value=0, max_value=1, value=1, step=1)

                    # Add a button to apply the selected tool
                    if st.button("Apply Tool", key="apply_button"):

                        # Check if an image is uploaded
                        if uploaded_file is None:
                            st.warning("Please upload an image before applying the tool.")
                            
                        else:
                            file_type = uploaded_file.type.split('/')[1]  # Get the file type (e.g., 'jpeg', 'png')
                            file_name = f"uploaded_image.{file_type}"  # Create a file name for the uploaded image

                            # Filtre fonksiyonu cagrilacak
                            # Apply filter adjustment

                            placeholder.empty()  # Clear the placeholder

                            enhanced_image = RobertsCrossOperator(image)

                            st.success(f"{tool} has been applied successfully. Check both of before-after images by slider!")
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1000)
                            
                            # ✅ Log to DB
                            # Log the process to the database
                            log_process(file_name, tool)
                                                  
                            # Dosyayı indirilebilir hale getirme
                            # Input field for custom file name
                            custom_file_name = st.text_input("Enter file name (without extension):", value="edited_image")

                            # Update the file name with the custom input
                            file_name = f"{custom_file_name}.{file_type}"

                            # Save the enhanced image to a BytesIO object
                            buffer = BytesIO()
                            enhanced_image.save(buffer, format=file_type.upper())
                            buffer.seek(0)

                            # Create a download button
                            st.download_button(
                                label="Download Edited Image",
                                data=buffer,
                                file_name=file_name,
                                mime=f"image/{file_type}",
                                key="download_button",
                                help="Click to download the edited image",
                                use_container_width=True  # Adjust the button width to fit the container
                            )


            case "Scharr Operator":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1000)
                    # Görseli Streamlit'te gösterelim
                    # Add a slider for filter values

                    dx = st.slider("dx", min_value=0, max_value=1, value=1, step=1)
                    dy = st.slider("dy", min_value=0, max_value=1, value=1, step=1)

                    # Add a button to apply the selected tool
                    if st.button("Apply Tool", key="apply_button"):

                        # Check if an image is uploaded
                        if uploaded_file is None:
                            st.warning("Please upload an image before applying the tool.")
                            
                        else:
                            file_type = uploaded_file.type.split('/')[1]  # Get the file type (e.g., 'jpeg', 'png')
                            file_name = f"uploaded_image.{file_type}"  # Create a file name for the uploaded image

                            # Filtre fonksiyonu cagrilacak
                            # Apply filter adjustment

                            placeholder.empty()  # Clear the placeholder

                            enhanced_image = ScharrOperator(image, int(dx), int(dy))

                            st.success(f"{tool} has been applied successfully. Check both of before-after images by slider!")
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1000)
                            
                            # ✅ Log to DB
                            # Log the process to the database
                            log_process(file_name, tool)
                            
                            # Dosyayı indirilebilir hale getirme
                            # Input field for custom file name
                            custom_file_name = st.text_input("Enter file name (without extension):", value="edited_image")

                            # Update the file name with the custom input
                            file_name = f"{custom_file_name}.{file_type}"

                            # Save the enhanced image to a BytesIO object
                            buffer = BytesIO()
                            enhanced_image.save(buffer, format=file_type.upper())
                            buffer.seek(0)

                            # Create a download button
                            st.download_button(
                                label="Download Edited Image",
                                data=buffer,
                                file_name=file_name,
                                mime=f"image/{file_type}",
                                key="download_button",
                                help="Click to download the edited image",
                                use_container_width=True  # Adjust the button width to fit the container
                            )

            case "Canny Edge Detection":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1000)
                    # Görseli Streamlit'te gösterelim
                    # Add a slider for filter values

                    low_threshold = st.slider("Low Threshold", min_value=0, max_value=255, value=75, step=1)
                    high_threshold = st.slider("High Threshold", min_value=0, max_value=255, value=175, step=1)

                    # Add a button to apply the selected tool
                    if st.button("Apply Tool", key="apply_button"):

                        # Check if an image is uploaded
                        if uploaded_file is None:
                            st.warning("Please upload an image before applying the tool.")
                            
                        else:
                            file_type = uploaded_file.type.split('/')[1]  # Get the file type (e.g., 'jpeg', 'png')
                            file_name = f"uploaded_image.{file_type}"  # Create a file name for the uploaded image

                            # Filtre fonksiyonu cagrilacak
                            # Apply filter adjustment

                            placeholder.empty()  # Clear the placeholder

                            enhanced_image = CannyEdgeDetection(image, int(low_threshold), int(high_threshold))

                            st.success(f"{tool} has been applied successfully. Check both of before-after images by slider!")
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1000)
                            
                            # ✅ Log to DB
                            # Log the process to the database
                            log_process(file_name, tool)
                   
                            st.success(f"{tool} has been applied successfully. Check both of before-after images by slider!")
                            
                            # Dosyayı indirilebilir hale getirme
                            # Input field for custom file name
                            custom_file_name = st.text_input("Enter file name (without extension):", value="edited_image")

                            # Update the file name with the custom input
                            file_name = f"{custom_file_name}.{file_type}"

                            # Save the enhanced image to a BytesIO object
                            buffer = BytesIO()
                            enhanced_image.save(buffer, format=file_type.upper())
                            buffer.seek(0)

                            # Create a download button
                            st.download_button(
                                label="Download Edited Image",
                                data=buffer,
                                file_name=file_name,
                                mime=f"image/{file_type}",
                                key="download_button",
                                help="Click to download the edited image",
                                use_container_width=True  # Adjust the button width to fit the container
                            )

    ##################################### FEATURE DETECTION ###############################

    if st.session_state.page == 'Feature':
        st.markdown("<h3 style='text-align: center; margin-top : 50px'>Feature Detection</h3>", unsafe_allow_html=True)

        st.markdown("<p style='font-size: 18px;'>Upload an image to apply the tool.</p>", unsafe_allow_html=True)

        placeholder = st.empty()

        # File uploader for image uploading
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

        # Görsel yüklendiyse, altta gösterelim
        if uploaded_file is not None:
            # Görseli açalım
            image = Image.open(uploaded_file).convert('L')
            
            placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1000)
            # Görseli Streamlit'te gösterelim
            # Add a slider for filter values

            # Add a button to apply the selected tool
            if st.button("Apply Tool", key="apply_button"):

                # Check if an image is uploaded
                if uploaded_file is None:
                    st.warning("Please upload an image before applying the tool.")
                    
                else:
                    file_type = uploaded_file.type.split('/')[1]
                    file_name = f"uploaded_image.{file_type}"  # Create a file name for the uploaded image

                    # Filtre fonksiyonu cagrilacak
                    # Apply filter adjustment

                    placeholder.empty()  # Clear the placeholder

                    mag1, dir1 = compute_gradients(np.array(image))


                    # # Anahtar noktaları bul
                    # kp1 = detect_keypoints_topk(mag1, top_k=50)
                    # kp2 = detect_keypoints_topk(mag2, top_k=50)

                    # Harris köşe tepkisini hesapla
                    R1 = harris_response(np.array(image))


                    # Anahtar noktaları bul
                    kp1 = detect_harris_keypoints(R1, threshold=0.001)

                    # Descriptor çıkar
                    desc1 = compute_descriptors(np.array(image), kp1)

                    # Draw keypoints
                    enhanced_image = draw_keypoints(np.array(image), kp1)

                    st.success("Feature Detection has been applied successfully. Check both of before-after images by slider!")
                    image_comparison(img1=image, img2=enhanced_image, label1="Original", label2="Feature Detection", width=1000)
                    
                    # ✅ Log to DB
                    # Log the process to the database
                    log_process(file_name, "Feature Detection")
                    
                    # Dosyayı indirilebilir hale getirme
                    # Input field for custom file name
                    custom_file_name = st.text_input("Enter file name (without extension):", value="edited_image")

                    # Update the file name with the custom input
                    file_name = f"{custom_file_name}.{file_type}"

                    # Save the enhanced image to a BytesIO object
                    buffer = BytesIO()
                    enhanced_image.save(buffer, format=file_type.upper())
                    buffer.seek(0)

                    # Create a download button
                    st.download_button(
                        label="Download Edited Image",
                        data=buffer,
                        file_name=file_name,
                        mime=f"image/{file_type}",
                        key="download_button",
                        help="Click to download the edited image",
                        use_container_width=True  # Adjust the button width to fit the container
                    )


    ################################### THRESHOLDING & SLICING #############################

    if st.session_state.page == 'Thresholding':
        st.markdown("<h3 style='text-align: center; margin-top : 50px'>Thresholding & Slicing</h3>", unsafe_allow_html=True)

        st.markdown(
            "<p style='font-size: 18px;'><br>Select a tool for thresholding and slicing.</p>",
        unsafe_allow_html=True
        )

        # Add a combobox for selecting a quick tool
        tool = st.selectbox(
            "Choose a tool:",
            ["Binary Thresholding", "Adaptive Thresholding", "Gray Level Slicing", "Bit Plane Slicing"]
        )

        # Display the selected tool
        st.markdown(f"<h3 style='margin-top: 3vh;'>{tool}</h3>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 18px;'>Upload an image to apply the selected tool.</p>", unsafe_allow_html=True)
        
        placeholder = st.empty()

        match tool:
            case "Binary Thresholding":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1000)
                    # Görseli Streamlit'te gösterelim
                    # Add a slider for filter values

                    threshold_value = st.slider("Threshold Value", min_value=0, max_value=255, value=127, step=1)

                    # Add a button to apply the selected tool
                    if st.button("Apply Tool", key="apply_button"):

                        # Check if an image is uploaded
                        if uploaded_file is None:
                            st.warning("Please upload an image before applying the tool.")
                            
                        else:
                            file_type = uploaded_file.type.split('/')[1]
                            file_name = f"uploaded_image.{file_type}"  # Create a file name for the uploaded image

                            # Filtre fonksiyonu cagrilacak
                            # Apply filter adjustment

                            placeholder.empty()  # Clear the placeholder

                            enhanced_image = BinaryThresholding(image, int(threshold_value))

                            st.success(f"{tool} has been applied successfully. Check both of before-after images by slider!")
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1000)
                            
                            # ✅ Log to DB
                            # Log the process to the database
                            log_process(file_name, tool)
                            
                            # Dosyayı indirilebilir hale getirme
                            # Input field for custom file name
                            custom_file_name = st.text_input("Enter file name (without extension):", value="edited_image")

                            # Update the file name with the custom input
                            file_name = f"{custom_file_name}.{file_type}"

                            # Save the enhanced image to a BytesIO object
                            buffer = BytesIO()
                            enhanced_image.save(buffer, format=file_type.upper())
                            buffer.seek(0)

                            # Create a download button
                            st.download_button(
                                label="Download Edited Image",
                                data=buffer,
                                file_name=file_name,
                                mime=f"image/{file_type}",
                                key="download_button",
                                help="Click to download the edited image",
                                use_container_width=True  # Adjust the button width to fit the container
                            )


            case "Adaptive Thresholding":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1000)
                    # Görseli Streamlit'te gösterelim
                    # Add a slider for filter values

                    max_value = st.slider("Max Value", min_value=0, max_value=255, value=255, step=1)
                    block_size = st.slider("Block Size", min_value=3, max_value=51, value=3, step=2) # 3-5-7-9-11-13-15
                    c = st.slider("C", min_value=-5, max_value=5, value=0, step=1)

                    adaptive_method = st.selectbox("Adaptive Method", ["cv2.ADAPTIVE_THRESH_MEAN_C", "cv2.ADAPTIVE_THRESH_GAUSSIAN_C"])
                    threshold_type = st.selectbox("Threshold Type", ["cv2.THRESH_BINARY", "cv2.THRESH_BINARY_INV"])


                    # Add a button to apply the selected tool
                    if st.button("Apply Tool", key="apply_button"):

                        # Check if an image is uploaded
                        if uploaded_file is None:
                            st.warning("Please upload an image before applying the tool.")
                            
                        else:
                            file_type = uploaded_file.type.split('/')[1]
                            file_name = f"uploaded_image.{file_type}"  # Create a file name for the uploaded image

                            # Filtre fonksiyonu cagrilacak
                            # Apply filter adjustment

                            placeholder.empty()  # Clear the placeholder

                            enhanced_image = AdaptiveThresholding(image, int(max_value), adaptive_method, threshold_type, int(block_size), int(c))

                            st.success(f"{tool} has been applied successfully. Check both of before-after images by slider!")                            
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1000)
                            
                            # ✅ Log to DB
                            # Log the process to the database
                            log_process(file_name, tool)
                            

                            # Dosyayı indirilebilir hale getirme
                            # Input field for custom file name
                            custom_file_name = st.text_input("Enter file name (without extension):", value="edited_image")

                            # Update the file name with the custom input
                            file_name = f"{custom_file_name}.{file_type}"

                            # Save the enhanced image to a BytesIO object
                            buffer = BytesIO()
                            enhanced_image.save(buffer, format=file_type.upper())
                            buffer.seek(0)

                            # Create a download button
                            st.download_button(
                                label="Download Edited Image",
                                data=buffer,
                                file_name=file_name,
                                mime=f"image/{file_type}",
                                key="download_button",
                                help="Click to download the edited image",
                                use_container_width=True  # Adjust the button width to fit the container
                            )


            case "Gray Level Slicing":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1000)
                    # Görseli Streamlit'te gösterelim
                    # Add a slider for filter values

                    min_gray = st.slider("Min Gray Level", min_value=0, max_value=255, value=80, step=1)
                    max_gray = st.slider("Max Gray Level", min_value=0, max_value=255, value=150, step=1)
                    highlight_value = st.slider("Highlight Value", min_value=0, max_value=255, value=255, step=1)

                    # Add a button to apply the selected tool
                    if st.button("Apply Tool", key="apply_button"):

                        # Check if an image is uploaded
                        if uploaded_file is None:
                            st.warning("Please upload an image before applying the tool.")
                            
                        else:
                            file_type = uploaded_file.type.split('/')[1]
                            file_name = f"uploaded_image.{file_type}"  # Create a file name for the uploaded image

                            # Filtre fonksiyonu cagrilacak
                            # Apply filter adjustment

                            placeholder.empty()  # Clear the placeholder

                            enhanced_image = GrayLevelSlicing(image, min_gray, max_gray, highlight_value)

                            st.success(f"{tool} has been applied successfully. Check both of before-after images by slider!")
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1000)
                            
                            # ✅ Log to DB
                            # Log the process to the database
                            log_process(file_name, tool)


                            # Dosyayı indirilebilir hale getirme
                            # Input field for custom file name
                            custom_file_name = st.text_input("Enter file name (without extension):", value="edited_image")

                            # Update the file name with the custom input
                            file_name = f"{custom_file_name}.{file_type}"

                            # Save the enhanced image to a BytesIO object
                            buffer = BytesIO()
                            enhanced_image.save(buffer, format=file_type.upper())
                            buffer.seek(0)

                            # Create a download button
                            st.download_button(
                                label="Download Edited Image",
                                data=buffer,
                                file_name=file_name,
                                mime=f"image/{file_type}",
                                key="download_button",
                                help="Click to download the edited image",
                                use_container_width=True  # Adjust the button width to fit the container
                            )


            case "Bit Plane Slicing":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1000)
                    # Görseli Streamlit'te gösterelim
                    # Add a slider for filter values

                    bit_plane = st.slider("Bit Plane", min_value=0, max_value=7, value=0, step=1) # 0-1-2-3-4-5-6-7

                    # Add a button to apply the selected tool
                    if st.button("Apply Tool", key="apply_button"):

                        # Check if an image is uploaded
                        if uploaded_file is None:
                            st.warning("Please upload an image before applying the tool.")
                            
                        else:
                            file_type = uploaded_file.type.split('/')[1]
                            file_name = f"uploaded_image.{file_type}"  # Create a file name for the uploaded image

                            # Filtre fonksiyonu cagrilacak
                            # Apply adjustment

                            placeholder.empty()  # Clear the placeholder

                            enhanced_image = BitPlaneSlicing(image, int(bit_plane))

                            st.success(f"{tool} has been applied successfully. Check both of before-after images by slider!")
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1000)
                            
                            # ✅ Log to DB
                            # Log the process to the database
                            log_process(file_name, tool)


                            # Dosyayı indirilebilir hale getirme
                            # Input field for custom file name
                            custom_file_name = st.text_input("Enter file name (without extension):", value="edited_image")

                            # Update the file name with the custom input
                            file_name = f"{custom_file_name}.{file_type}"

                            # Save the enhanced image to a BytesIO object
                            buffer = BytesIO()
                            enhanced_image.save(buffer, format=file_type.upper())
                            buffer.seek(0)

                            # Create a download button
                            st.download_button(
                                label="Download Edited Image",
                                data=buffer,
                                file_name=file_name,
                                mime=f"image/{file_type}",
                                key="download_button",
                                help="Click to download the edited image",
                                use_container_width=True  # Adjust the button width to fit the container
                            )

    if st.session_state.page == "Database":

        st.markdown("<h3 style='text-align: center; margin-top : 50px'>Database</h3>", unsafe_allow_html=True)

        st.markdown(
            "<p style='font-size: 18px;'><br>The database contains the file processes as history.</p>",
        unsafe_allow_html=True
        )

        # Display database content as a table
        st.markdown("<h5 style='text-align: center; margin-top : 20px'>Process History Table</h5>", unsafe_allow_html=True)

        logs = get_logs()
        df_logs = pd.DataFrame(logs, columns=["Tarih", "Saat", "Dosya Adı", "Kullanılan Metot"])
        st.dataframe(df_logs, use_container_width=True)


    if st.session_state.page == 'About':

        st.markdown("<h3 style='text-align: center; margin-top : 50px'>About</h3>", unsafe_allow_html=True)

        st.markdown(
            "<p style='font-size: 18px;'><br>This is a simple image processing web application built using Streamlit.</p>",
        unsafe_allow_html=True
        )

        st.markdown(
            "<p style='font-size: 18px;'>It provides various image processing tools such as:</p>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<ul style='font-size: 18px;'>"
            "<li>Image comparison by slider</li>"
            "<li>Filtering</li>"
            "<li>Edge detection</li>"
            "<li>Thresholding</li>"
            "<li>Slicing</li>"
            "<li>Image enhancement</li>"
            "<li>and others.</li>"
            "</ul>",
            unsafe_allow_html=True
        )


############################# COL3 - DIP TECHNIQUES' INFORMATION PART #############################

with col3:

    if st.session_state.page != 'Dashboard':
        # Add a gray box with "Technique Info" header and a thin underline

        if st.session_state.page == 'Contrast':
            
            if tool == "Saturation":
                st.markdown("""
                    <div style='background-color: #e01410; padding: 20px; border-radius: 10px; margin-top: 20px;  height: 80vh;'>
                        <h3 style='margin: 0; color: #FFF;'>Saturation</h3>
                        <hr style='border: none; border-top: 1px solid #FFF; margin: 10px 0;'>
                        <p style='font-size: 16px; color: #FFF;'>Saturation controls the intensity and vividness of colors in an image. 
                        It determines how "pure" or "washed out" a color appears.</p>
                        <p style='font-size: 16px; color: #FFF;'>Saturation Value:
                        <ul style='font-size: 16px; color: #FFF;'>
                            <li>= 0 → removes all color, turning the image into grayscale.</li>
                            <li>= 1 → keeps the original saturation of the image.</li>
                            <li>&gt; 1 → enhances color intensity, making the image appear more lively and dramatic.</li>
                        </ul>
                        <p style='font-size: 16px; color: #FFF;'>Adjusting saturation is commonly used to correct dull images, apply creative effects, or match a specific visual style. 
                        It's a powerful tool in both photography and digital image processing.</p>
                    </div>
                """, unsafe_allow_html=True)

            elif tool == "Histogram Equalization":

                st.markdown("""
                    <div style='background-color: #e01410; padding: 20px; border-radius: 10px; margin-top: 20px;  height: 80vh;'>
                        <h3 style='margin: 0; color: #FFF;'>Histogram Equalization</h3>
                        <hr style='border: none; border-top: 1px solid #FFF; margin: 10px 0;'>
                        <p style='font-size: 16px; color: #FFF;'>
                            Histogram Equalization is a technique used to improve the contrast of an image by redistributing the intensity values. 
                            It spreads out the most frequent intensity values, enhancing the global contrast.
                        </p>
                        <ul style='font-size: 16px; color: #FFF;'>
                            <li>Ideal for low contrast images.</li>
                            <li>Enhances details in both dark and bright regions.</li>
                            <li>Works well in grayscale images.</li>
                        </ul>
                        <p style='font-size: 16px; color: #FFF;'>Commonly used in medical imaging, satellite imagery, and photography to improve visibility of features.</p>
                    </div>
                """, unsafe_allow_html=True)


            elif tool == "Gamma Correction":

                st.markdown("""
                    <div style='background-color: #e01410; padding: 20px; border-radius: 10px; margin-top: 20px;  height: 80vh;'>
                        <h3 style='margin: 0; color: #FFF;'>Gamma Correction</h3>
                        <hr style='border: none; border-top: 1px solid #FFF; margin: 10px 0;'>
                        <p style='font-size: 16px; color: #FFF;'>
                            Gamma Correction adjusts the brightness of an image by applying a nonlinear transformation to pixel values.
                        </p>
                        <ul style='font-size: 16px; color: #FFF;'>
                            <li>Gamma &gt; 1 → Makes the image brighter.</li>
                            <li>Gamma = 1 → No change to brightness.</li>
                            <li>Gamma &lt; 1 → Makes the image darker.</li>
                        </ul>
                        <p style='font-size: 16px; color: #FFF;'>Useful for correcting lighting issues, improving visibility, or preparing images for display devices.</p>
                    </div>
                """, unsafe_allow_html=True)


        elif st.session_state.page == "Blurring":

            if tool == "Gaussian Blurring":

                st.markdown("""
                    <div style='background-color: #e01410; padding: 20px; border-radius: 10px; margin-top: 20px;  height: 80vh;'>
                        <h3 style='margin: 0; color: #FFF;'>Gaussian Blurring</h3>
                        <hr style='border: none; border-top: 1px solid #FFF; margin: 10px 0;'>
                        <p style='font-size: 16px; color: #FFF;'>
                            Gaussian Blurring is a smoothing technique that uses a Gaussian function to reduce image noise and detail. 
                            It is widely used before edge detection and other preprocessing steps.
                        </p>
                        <ul style='font-size: 16px; color: #FFF;'>
                            <li>Applies a weighted average where nearby pixels influence the center more.</li>
                            <li>Preserves general structures while removing fine noise.</li>
                            <li>Controlled by kernel size and standard deviation (sigma).</li>
                        </ul>
                        <p style='font-size: 16px; color: #FFF;'>Commonly used in object detection, segmentation, and background blurring.</p>
                    </div>
                """, unsafe_allow_html=True)

            
            elif tool == "Median Blurring":

                st.markdown("""
                    <div style='background-color: #e01410; padding: 20px; border-radius: 10px; margin-top: 20px;  height: 80vh;'>
                        <h3 style='margin: 0; color: #FFF;'>Median Blurring</h3>
                        <hr style='border: none; border-top: 1px solid #FFF; margin: 10px 0;'>
                        <p style='font-size: 16px; color: #FFF;'>
                            Median Blurring replaces each pixel's value with the median of neighboring pixels. 
                            It is particularly effective for removing "salt and pepper" noise.
                        </p>
                        <ul style='font-size: 16px; color: #FFF;'>
                            <li>Preserves edges better than other blurring techniques.</li>
                            <li>Ideal for images with high-contrast noise.</li>
                            <li>Kernel size must be an odd number (e.g., 3, 5, 7).</li>
                        </ul>
                        <p style='font-size: 16px; color: #FFF;'>Widely used in preprocessing steps for cleaner edge and object detection.</p>
                    </div>
                """, unsafe_allow_html=True)


            elif tool == "Average Blurring":

                st.markdown("""
                    <div style='background-color: #e01410; padding: 20px; border-radius: 10px; margin-top: 20px;  height: 80vh;'>
                        <h3 style='margin: 0; color: #FFF;'>Average Blurring</h3>
                        <hr style='border: none; border-top: 1px solid #FFF; margin: 10px 0;'>
                        <p style='font-size: 16px; color: #FFF;'>
                            Average Blurring (or mean filtering) smoothens an image by replacing each pixel with the average of its neighboring pixels.
                        </p>
                        <ul style='font-size: 16px; color: #FFF;'>
                            <li>Reduces noise and detail uniformly.</li>
                            <li>Can blur edges and textures.</li>
                            <li>Simple and fast technique using a normalized box filter.</li>
                        </ul>
                        <p style='font-size: 16px; color: #FFF;'>Useful when performance matters and edge precision is less critical.</p>
                    </div>
                """, unsafe_allow_html=True)


            elif tool == "Bilateral Filtering":

                st.markdown("""
                    <div style='background-color: #e01410; padding: 20px; border-radius: 10px; margin-top: 20px;  height: 80vh;'>
                        <h3 style='margin: 0; color: #FFF;'>Bilateral Filtering</h3>
                        <hr style='border: none; border-top: 1px solid #FFF; margin: 10px 0;'>
                        <p style='font-size: 16px; color: #FFF;'>
                            Bilateral Filtering smoothens images while preserving edges. It considers both spatial distance and pixel intensity difference.
                        </p>
                        <ul style='font-size: 16px; color: #FFF;'>
                            <li>Preserves sharp edges while reducing noise.</li>
                            <li>Controlled by three parameters: diameter (d), sigmaColor, and sigmaSpace.</li>
                            <li>Higher sigmaColor → more colors are mixed; Higher sigmaSpace → larger areas are blurred.</li>
                        </ul>
                        <p style='font-size: 16px; color: #FFF;'>Ideal for edge-preserving denoising and cartoon-like effects.</p>
                    </div>
                """, unsafe_allow_html=True)



        elif st.session_state.page == "Edge":
            
            if tool == "SOBEL Operator":
                st.markdown("""
                    <div style='background-color: #e01410; padding: 20px; border-radius: 10px; margin-top: 20px;  height: 80vh;'>
                        <h3 style='margin: 0; color: #FFF;'>SOBEL Operator</h3>
                        <hr style='border: none; border-top: 1px solid #FFF; margin: 10px 0;'>
                        <p style='font-size: 16px; color: #FFF;'>
                            The Sobel Operator detects edges by calculating the gradient of image intensity in horizontal and vertical directions.
                        </p>
                        <ul style='font-size: 16px; color: #FFF;'>
                            <li>Highlights edges with strong intensity changes.</li>
                            <li>Combines smoothing and differentiation.</li>
                            <li>Commonly applied in both x and y directions.</li>
                        </ul>
                        <p style='font-size: 16px; color: #FFF;'>Widely used for object boundary detection and edge-based segmentation.</p>
                    </div>
                """, unsafe_allow_html=True)


            elif tool == "Laplacian of Gaussian (LoG)":
                
                st.markdown("""
                    <div style='background-color: #e01410; padding: 20px; border-radius: 10px; margin-top: 20px;  height: 80vh;'>
                        <h3 style='margin: 0; color: #FFF;'>Laplacian of Gaussian (LoG)</h3>
                        <hr style='border: none; border-top: 1px solid #FFF; margin: 10px 0;'>
                        <p style='font-size: 16px; color: #FFF;'>
                            Laplacian of Gaussian combines Gaussian smoothing and Laplacian edge detection. It helps detect edges while reducing noise.
                        </p>
                        <ul style='font-size: 16px; color: #FFF;'>
                            <li>First applies Gaussian blur to reduce noise.</li>
                            <li>Then uses Laplacian operator to detect regions with rapid intensity change.</li>
                            <li>Effective in highlighting fine details and edges.</li>
                        </ul>
                        <p style='font-size: 16px; color: #FFF;'>Ideal for edge detection in noisy images and medical/scientific analysis.</p>
                    </div>
                """, unsafe_allow_html=True)


            elif tool == "Prewitt Operator":
                
                st.markdown("""
                    <div style='background-color: #e01410; padding: 20px; border-radius: 10px; margin-top: 20px;  height: 80vh;'>
                        <h3 style='margin: 0; color: #FFF;'>Prewitt Operator</h3>
                        <hr style='border: none; border-top: 1px solid #FFF; margin: 10px 0;'>
                        <p style='font-size: 16px; color: #FFF;'>
                            The Prewitt Operator detects edges by computing the gradient of the image using a simpler kernel than Sobel.
                        </p>
                        <ul style='font-size: 16px; color: #FFF;'>
                            <li>Focuses on horizontal and vertical intensity changes.</li>
                            <li>Less sensitive to noise than Laplacian-based methods.</li>
                            <li>Simple and computationally efficient.</li>
                        </ul>
                        <p style='font-size: 16px; color: #FFF;'>Used in basic edge detection tasks where performance is crucial.</p>
                    </div>
                """, unsafe_allow_html=True)


            elif tool == "Roberts Cross Operator":
                
                st.markdown("""
                    <div style='background-color: #e01410; padding: 20px; border-radius: 10px; margin-top: 20px;  height: 80vh;'>
                        <h3 style='margin: 0; color: #FFF;'>Roberts Cross Operator</h3>
                        <hr style='border: none; border-top: 1px solid #FFF; margin: 10px 0;'>
                        <p style='font-size: 16px; color: #FFF;'>
                            The Roberts Cross Operator is a simple method for edge detection using a diagonal gradient approximation.
                        </p>
                        <ul style='font-size: 16px; color: #FFF;'>
                            <li>Uses 2×2 kernels for diagonal edge detection.</li>
                            <li>Effective for high-speed processing with minimal computation.</li>
                            <li>Sensitive to noise due to small kernel size.</li>
                        </ul>
                        <p style='font-size: 16px; color: #FFF;'>Often used in real-time applications or where memory is limited.</p>
                    </div>
                """, unsafe_allow_html=True)


            elif tool == "Scharr Operator":

                st.markdown("""
                    <div style='background-color: #e01410; padding: 20px; border-radius: 10px; margin-top: 20px;  height: 80vh;'>
                        <h3 style='margin: 0; color: #FFF;'>Scharr Operator</h3>
                        <hr style='border: none; border-top: 1px solid #FFF; margin: 10px 0;'>
                        <p style='font-size: 16px; color: #FFF;'>
                            The Scharr Operator is an improved version of Sobel, designed for better rotational symmetry and edge accuracy.
                        </p>
                        <ul style='font-size: 16px; color: #FFF;'>
                            <li>Emphasizes gradient accuracy, especially for diagonal edges.</li>
                            <li>Uses more precise kernel weights than Sobel.</li>
                            <li>Suitable for high-quality edge detection.</li>
                        </ul>
                        <p style='font-size: 16px; color: #FFF;'>Common in applications requiring accurate and detailed edge maps.</p>
                    </div>
                """, unsafe_allow_html=True)


            elif tool == "Canny Edge Detection":
                
                st.markdown("""
                    <div style='background-color: #e01410; padding: 20px; border-radius: 10px; margin-top: 20px;  height: 80vh;'>
                        <h3 style='margin: 0; color: #FFF;'>Canny Edge Detection</h3>
                        <hr style='border: none; border-top: 1px solid #FFF; margin: 10px 0;'>
                        <p style='font-size: 16px; color: #FFF;'>
                            Canny Edge Detection is a multi-stage algorithm for detecting sharp edges in images with high accuracy.
                        </p>
                        <ul style='font-size: 16px; color: #FFF;'>
                            <li>Applies Gaussian blur to reduce noise.</li>
                            <li>Finds intensity gradients and suppresses non-maximum edges.</li>
                            <li>Uses double thresholding and edge tracking for precision.</li>
                        </ul>
                        <p style='font-size: 16px; color: #FFF;'>Widely used in computer vision and object detection tasks due to its robustness and accuracy.</p>
                    </div>
                """, unsafe_allow_html=True)


        elif st.session_state.page == "Thresholding":

            if tool == "Binary Thresholding":
                
                st.markdown("""
                    <div style='background-color: #e01410; padding: 20px; border-radius: 10px; margin-top: 20px;  height: 80vh;'>
                        <h3 style='margin: 0; color: #FFF;'>Binary Thresholding</h3>
                        <hr style='border: none; border-top: 1px solid #FFF; margin: 10px 0;'>
                        <p style='font-size: 16px; color: #FFF;'>
                            Binary Thresholding converts a grayscale image into black and white based on a fixed threshold value.
                        </p>
                        <ul style='font-size: 16px; color: #FFF;'>
                            <li>Pixels above the threshold → set to max value (usually 255).</li>
                            <li>Pixels below the threshold → set to 0.</li>
                            <li>Fast and simple segmentation method.</li>
                        </ul>
                        <p style='font-size: 16px; color: #FFF;'>Commonly used for separating foreground from background in document processing or shape analysis.</p>
                    </div>
                """, unsafe_allow_html=True)


            elif tool == "Adaptive Thresholding":
                
                st.markdown("""
                    <div style='background-color: #e01410; padding: 20px; border-radius: 10px; margin-top: 20px;  height: 80vh;'>
                        <h3 style='margin: 0; color: #FFF;'>Binary Thresholding</h3>
                        <hr style='border: none; border-top: 1px solid #FFF; margin: 10px 0;'>
                        <p style='font-size: 16px; color: #FFF;'>
                            Binary Thresholding converts a grayscale image into black and white based on a fixed threshold value.
                        </p>
                        <ul style='font-size: 16px; color: #FFF;'>
                            <li>Pixels above the threshold → set to max value (usually 255).</li>
                            <li>Pixels below the threshold → set to 0.</li>
                            <li>Fast and simple segmentation method.</li>
                        </ul>
                        <p style='font-size: 16px; color: #FFF;'>Commonly used for separating foreground from background in document processing or shape analysis.</p>
                    </div>
                """, unsafe_allow_html=True)


            elif tool == "Gray Level Slicing":
                
                st.markdown("""
                    <div style='background-color: #e01410; padding: 20px; border-radius: 10px; margin-top: 20px;  height: 80vh;'>
                        <h3 style='margin: 0; color: #FFF;'>Gray Level Slicing</h3>
                        <hr style='border: none; border-top: 1px solid #FFF; margin: 10px 0;'>
                        <p style='font-size: 16px; color: #FFF;'>
                            Gray Level Slicing enhances specific ranges of pixel intensities while suppressing others.
                        </p>
                        <ul style='font-size: 16px; color: #FFF;'>
                            <li>Highlights certain intensity levels to emphasize features.</li>
                            <li>Can be applied with or without preserving the rest of the image.</li>
                            <li>Useful for isolating regions of interest (e.g., bones in X-rays).</li>
                        </ul>
                        <p style='font-size: 16px; color: #FFF;'>Great for visualizing specific structures in scientific or medical imagery.</p>
                    </div>
                """, unsafe_allow_html=True)


            elif tool == "Bit Plane Slicing":

                st.markdown("""
                    <div style='background-color: #e01410; padding: 20px; border-radius: 10px; margin-top: 20px;  height: 80vh;'>
                        <h3 style='margin: 0; color: #FFF;'>Bit Plane Slicing</h3>
                        <hr style='border: none; border-top: 1px solid #FFF; margin: 10px 0;'>
                        <p style='font-size: 16px; color: #FFF;'>
                            Bit Plane Slicing separates an image into its individual bit planes, highlighting the contribution of each bit to the image.
                        </p>
                        <ul style='font-size: 16px; color: #FFF;'>
                            <li>Each 8-bit grayscale image contains 8 bit planes (from bit 0 to bit 7).</li>
                            <li>Higher-order bits (e.g., 6th, 7th) contribute more to the visual content.</li>
                            <li>Lower-order bits often contain finer details or noise.</li>
                        </ul>
                        <p style='font-size: 16px; color: #FFF;'>Useful in image compression, watermarking, and feature extraction.</p>
                    </div>
                """, unsafe_allow_html=True)
