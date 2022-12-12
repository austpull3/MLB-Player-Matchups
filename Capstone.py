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
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    img = plt.imread("Images/braves.png")
    fig = plt.figure(figsize = (20,20))
    fig, ax = plt.subplots()
    ax.imshow(img, extent = [-100,500,385,-50])
    ax.scatter(a.hc_x, a.hc_y)
    ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
    # disabling xticks by Setting xticks to an empty list
    plt.xticks([]) 

    # disabling yticks by setting yticks to an empty list
    plt.yticks([]) 
    st.pyplot(fig)
    
    img = plt.imread("Images/diamond.png")
    fig = plt.figure(figsize = (23,20))
    fig, ax = plt.subplots()
    ax.imshow(img, extent = [-100,500,385,-50])
    ax.scatter(a.hc_x, a.hc_y)
    ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
    # disabling xticks by Setting xticks to an empty list
    plt.xticks([]) 

    # disabling yticks by setting yticks to an empty list
    plt.yticks([]) 
    st.pyplot(fig)
    
    from PIL import Image 
    image7 = Image.open('Images/dansby.png')
    st.image(image7)
    
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    img = plt.imread("Images/dansby.png")
    fig = plt.figure(figsize = (20,20))
    fig, ax = plt.subplots()
    ax.imshow(img, extent = [-100,500,385,-50])
    ax.scatter(a.hc_x, a.hc_y)
    ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
    plt.show()
    st.pyplot(fig)

    
    
    
    

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
    add_bg_from_local('Images/blue.webp')
    
    league = st.radio("Select National League or American League:", ("NL", "AL"))
    if league == "NL":
        st.write("You selected the National League.")
        nlteam = st.radio("Select a National League Team.", ('Braves', 'Marlins', 'Mets', 'Nationals', ' Phillies', 'Brewers', 'Cardinals', 'Cubs', 'Pirates', 'Reds', 'D-backs', 'Dodgers', 'Giants', 'Padres', 'Rockies'))
        if nlteam == "Braves":
            col4, col5, col6 = st.columns(3)
            with col5:
                from PIL import Image 
                braves = Image.open('Images/braves logo.png')
                st.image(braves)
                st.write("")
                st.write(" ")
    else:
        alteam = st.radio("Select an American League Team.", ('Blue Jays', 'Orioles', 'Rays', 'Red Sox', 'Yankees', 'Guardians', 'Royals', 'Tigers', 'Twins', 'White Sox', 'Angels', 'Astros', 'Athletics', 'Mariners', 'Rangers'))
        if alteam == "Blue Jays":
            hitter = st.selectbox("Please select a hitter.", ['Bo Bichette: TOR (SS/R)', 'Vladimir Guerrero Jr.: TOR (1B/R)', 'Teoscar Hernandez: TOR (RF/R)'])
            if hitter == "Bo Bichette":
                hitter = 666182
            elif hitter == "Vladimir Guerrero Jr.":
                hitter = 665489
            else:
                hitter = 606192
        pitcherdiv = st.radio("Select pitcher from the AL or NL.", ('AL', 'NL'))
        if pitcherdiv == 'AL':
               pitcheral = st.selectbox("Please select an AL pitcher.", ['Tyler Wells: BAL (R)', 'Drew Rasmussen: TB (R)', 'Nick Pivetta: BOS (R)', 'Nestor Cortes: NYY (L)', 'Triston McKenzie: CLE (R)', 'Brady Singer: KC (R)', 'Tarik Skubal: DET (L)', 'Sonny Gray: MIN (R)', 'Dylan Cease: CWS (R)', 'Shohei Ohtani: LAA (R)', 'Justin Verlander: HOU (R)', 'James Kaprielian: OAK (R)', 'Chris Flexen: SEA (R)', 'Martin Perez: TEX (L)'])
        else:
           pitchernl = st.selectbox("Please select an NL pitcher.", ['Max Fried: ATL (L)', 'Sandy Alcantara: MIA (R)', 'Max Scherzer: NYM (R)', 'Paolo Espino: WSH (R)', 'Aaron Nola: PHI (R)', 'Corbin Burnes: MIL (R)', 'Miles Mikolas: STL (R)', 'Marcus Stroman: CHC (R)', 'Jose Quintana: PIT (L)', 'Nick Lodolo: CIN (L)', 'Zac Gallen: ARI (R)', 'Clayton Kershaw: LAD (L)', 'Carlos Rodon: SF (L)', 'Blake Snell: SD (L)', 'Ryan Feltner: COL (R)'])
        
page_names_to_funcs = {
    "Welcome Page": main_page,
    "At-Bat Predictor": page2
    }

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
