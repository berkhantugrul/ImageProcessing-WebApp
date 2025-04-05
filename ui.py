import streamlit as st
from PIL import Image
from pathlib import Path
from streamlit_image_comparison import image_comparison
from image_processes import *
from wordcloud import WordCloud
import matplotlib.pyplot as plt


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

if st.sidebar.button('Edge & Feature Detection'):
    st.session_state.page = 'Edge'

if st.sidebar.button('Thresholding & Slicing'):
    st.session_state.page = 'Thresholding'

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

        # Display two cards side by side
        st.markdown("""
            <div style='display: flex; justify-content: flex-start; gap: 20px; margin-top: 40px; margin-left: 0px; width: 120%;'>
            <div style='display: flex; align-items: center; justify-content: center; background-color: #f0f0f0; width: 45vh; height: 25vh; border-radius: 10px; padding: 20px;'>
            <div style='width: 80px; height: 8vh; border: 2px solid transparent; border-radius: 50%; display: flex; justify-content: center; align-items: center; background-color: #f0f0f0; margin-right: 50px; margin-left: 10px;'>
            <img src='https://cdn-icons-png.freepik.com/128/9752/9752076.png' alt='Icon' style='width: 60px; height: 60px;' />
            </div>
            <div style='text-align: left;'>
            <h4 style='margin-top: 15px; color: #000000'>Total:</h4>
            <h3 style='margin-top: 5px; color: #dd140e;'>17 filters</h3>
            <p style='margin-bottom: 40px; color: #000000'>have been implemented to that project.</p>
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
            <img src='https://cdn-icons-png.freepik.com/128/1159/1159633.png' alt='Icon' style='width: 60px; height: 60px;' />
            </div>
            <div style='text-align: left;'>
            <h4 style='margin-top: 15px; color: #000000'>Total:</h4>
            <h3 style='margin-top: 5px; color: #dd140e;'>0 image(s)</h3>
            <p style='margin-bottom: 40px; color: #000000'>have been edited in this project.</p>
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
            "Blurring": 6
        }

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
                
                placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1170)
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
                        image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1170)

                        
                        # Dosyayı indirilebilir hale getirme
                        # Input field for custom file name
                        custom_file_name = st.text_input("Enter file name (without extension):", value="edited_image")

                        # Update the file name with the custom input
                        file_name = f"{custom_file_name}.{file_type}"

                        
                        
        elif tool == "Histogram Equalization":
            # File uploader for image uploading
            
            uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

            # Görsel yüklendiyse, altta gösterelim
            if uploaded_file is not None:
                # Görseli açalım
                image = Image.open(uploaded_file)
                
                placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1170)
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
                        image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1170)

                        
                        # Dosyayı indirilebilir hale getirme
                        # Input field for custom file name
                        custom_file_name = st.text_input("Enter file name (without extension):", value="edited_image")

                        # Update the file name with the custom input
                        file_name = f"{custom_file_name}.{file_type}"



        elif tool == "Gamma Correction":

            # File uploader for image uploading
            
            uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

            # Görsel yüklendiyse, altta gösterelim
            if uploaded_file is not None:
                # Görseli açalım
                image = Image.open(uploaded_file)
                
                placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1170)
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
                        image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1170)

                        
                        # Dosyayı indirilebilir hale getirme
                        # Input field for custom file name
                        custom_file_name = st.text_input("Enter file name (without extension):", value="edited_image")

                        # Update the file name with the custom input
                        file_name = f"{custom_file_name}.{file_type}"


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
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1170)
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
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1170)

                            
                            


            case "Median Blurring":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1170)
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
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1170)

                            
                            

        
            case "Average Blurring":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1170)
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
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1170)

                            
                            

                
            case "Bilateral Filtering":
                # File uploader for image uploading
                
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1170)
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
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1170)

                            
    ################################## EDGE & FEATURE DETECTION #############################

            
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
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1170)
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
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1170)


            case "Laplacian of Gaussian (LoG)":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1170)
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
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1170)


            case "Prewitt Operator":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1170)
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
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1170)



            case "Roberts Cross Operator":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1170)
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
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1170)


            case "Scharr Operator":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1170)
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
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1170)


            case "Canny Edge Detection":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1170)
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
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1170)



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
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1170)
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
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1170)


            case "Adaptive Thresholding":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1170)
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
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1170)

                            st.success(f"{tool} has been applied successfully. Check both of before-after images by slider!")


            case "Gray Level Slicing":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1170)
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
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1170)

                            st.success(f"{tool} has been applied successfully. Check both of before-after images by slider!")


            case "Bit Plane Slicing":
                # File uploader for image uploading
                uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

                # Görsel yüklendiyse, altta gösterelim
                if uploaded_file is not None:
                    # Görseli açalım
                    image = Image.open(uploaded_file)
                    
                    placeholder.image(image, caption=f"Image dimensions: {image.size[0]} x {image.size[1]} (Width x Height)", width=1170)
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
                            image_comparison(img1=image, img2=enhanced_image, label1="Original", label2=f"{tool}", width=1170)

                            st.success(f"{tool} has been applied successfully. Check both of before-after images by slider!")



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

        
with col3:

    if st.session_state.page != 'Dashboard':
        # Add a gray box with "Technique Info" header and a thin underline
        st.markdown("""
            <div style='background-color: #e01410; padding: 20px; border-radius: 10px; margin-top: 20px; height: 100vh;'>
                <h3 style='margin: 0; color: #FFF;'>Technique Info</h3>
                <hr style='border: none; border-top: 1px solid #FFF; margin: 10px 0;'>
                <p style='font-size: 16px; color: #FFF;'>Detailed information about the selected technique will be displayed here.</p>
            </div>
        """, unsafe_allow_html=True)


    # ilgili kısım eklenecek