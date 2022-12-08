import streamlit as st

st.title("Predict MLB At-Bat Outcomes⚾️")

def main_page():    
    from PIL import Image 
    image1 = Image.open('mlb.png')
    st.image(image1)

    st.sidebar.markdown("# Welcome!⚾️")
    st.sidebar.markdown(" ")
    if st.sidebar.checkbox(" Select For Help 🔍"):
        st.sidebar.info("This is the welcome page which describes how to interact with the different pages and the purpose of the Streamlit app.")
        st.sidebar.markdown("### Above ⬆ is a drop down of different pages to navigate through. Select the page you are interested in exploring.")
        
        
        
        
page_names_to_funcs = {
    "Welcome Page": main_page,
    
    }

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()


