import streamlit as st

st.title("Predict MLB At-Bat Outcomes ⚾️") 

def main_page():    
    from PIL import Image 
    image1 = Image.open('Images/mlb.png')
    st.image(image1, caption = "https://en.wikipedia.org/wiki/Major_League_Baseball_logo")
    image3 = Image.open('Images/judgehitting.jpeg')
    image4 = Image.open('Images/justinv.jpeg')
    image5 = Image.open('Images/vs.jpeg')
    st.markdown("##")
    st.markdown("### Explore the predicted at-bat outcomes of the games best!")
    col, col2, col3 = st.columns(3)
   
    with col:
        st.image(image3) 
    with col2:
        st.image(image5)
    with col3:
        st.image(image4)

    st.sidebar.markdown("# Welcome!⚾️")
    st.sidebar.markdown(" ")
    if st.sidebar.checkbox(" Select For Help 🔍"):
        st.sidebar.info("This is the welcome page which describes how to interact with the different pages and the purpose of the Streamlit app.")
        st.sidebar.markdown("### Above ⬆ is a drop down of different pages to navigate through. Select the page you are interested in exploring.")
        
    
        #Import necessary libraries
    import pandas as pd
    import plotly.graph_objects as go
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    import io
    df = pd.read_csv("bh2.csv") 
    st.write(df.head())
    
    from pybaseball import statcast
    data = pd.read_csv("dans.csv")
    st.write(data.head())
    from pybaseball import statcast_batter, spraychart
    a = statcast_batter('2022-04-07', '2022-10-02', 621020)
    st.write(a.events.value_counts())
    a = a[a['pitcher']== 453286]
    spraychart(a, 'braves', title='Dansby Swanson vs Max Scherser', colorby='player')
    from pybaseball import playerid_lookup
    if st.selectbox("Select player", ['Dansby Swanson']):
        st.write(playerid_lookup('Swanson', 'Dansby'))
        import warnings
        warnings.filterwarnings('ignore')
        spraychart(a, 'braves', title='Dansby Swanson vs Max Scherzer (2022)', colorby='events')
        plt.show()
    
    
    
    

def page2():
    import base64
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
                background-size: cover
            }}
            </style>
            """,
            unsafe_allow_html=True
            )
    add_bg_from_local('Images/tmobile.jpeg')

   
        
page_names_to_funcs = {
    "Welcome Page": main_page,
    "At-Bat Predictor": page2
    }

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
