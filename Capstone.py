#import necessary libraries
import streamlit as st
import plotly.express as px
from pybaseball import playerid_lookup
from pybaseball import statcast_batter, spraychart
from pybaseball import batting_stats
from pybaseball import pitching_stats
from pybaseball import statcast
import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure 
import matplotlib.pyplot as plt
import io
from streamlit_player import st_player
import requests

st.set_page_config(page_icon=":baseball:", page_title="MLB At-Bat Predictor") 
#create first page/welcome page
def main_page():    
    st.title("Welcome to the MLB At-Bat Predictor")
    #load images for first page with justin verlander and aaron judge
    from PIL import Image 
    image1 = Image.open('Images/mlb.png')
    st.image(image1, caption = "https://en.wikipedia.org/wiki/Major_League_Baseball_logo")
    image3 = Image.open('Images/judgehitting.jpeg')
    image4 = Image.open('Images/justinv.jpeg')
    image5 = Image.open('Images/vs.jpeg') #versus image
    st.markdown("##")
    st.markdown("### Explore the predicted at-bat outcomes of the games best!")
    #columns for having a good display of the images
    col, col2, col3 = st.columns(3)
   
    with col:
        st.image(image3) #judge
    with col2:
        st.image(image5) #versus image
    with col3:
        st.image(image4) #verlander
        
    st.write(" ")
    st.markdown("### Do you want to predict moments like this? Continue to the other pages.")
    #st_player("https://www.youtube.com/watch?v=clDXWm1jpfY") #include youtube video 
    #st.write("Source: https://www.youtube.com/watch?v=clDXWm1jpfY") #link to the same video
    
    #sidebar instructions and layout
    st.sidebar.markdown("# Welcome!⚾️")
    st.sidebar.markdown(" ")
    if st.sidebar.checkbox(" Select For Help ⚾️"):
        st.sidebar.info("Welcome to the MLB At-Bat Outcome prediction application. To explore player outcome data and display some spraycharts go to the next page. If you want to predict at-bat outcomes go to the last page.")
        st.sidebar.info("Play the video at the bottom of the page to see an exciting at-bat.")
        st.sidebar.markdown("### The drop down above ↑ includes different pages to navigate through. Select the next page to explore MLB data or the last page to make predictions. Enjoy!")
 

    
    
    #Module testing
    def test_youtube_video():

        # Test if the video is displayed in Streamlit
        assert st_player("https://www.youtube.com/watch?v=clDXWm1jpfY", st.write("This is the actual information hereheheheheh")), "Fail!!!"

        # Test if the link to the video is displayed
        assert st.write("https://www.youtube.com/watch?v=clDXWm1jpfY"), "Failed to display link to video"
    test_youtube_video()



#define page 2 for visualizing player spraycharts    
def page2():
    st.title("Explore MLB Data and Visualize Spraycharts of your Favorite Players ⚾️") 
    st.markdown("#### Enter players from the 2022 season only!") #only 2022 players should be entered
    #try code for entering player names and displaying a spray chart plot
    try: 
        first_name = st.text_input('Enter a players first name:')
        first_name = first_name.strip()
        if " " in first_name:
            st.error("Please do not include whitespace in the input.") #address error
        if first_name.isspace():
            st.warning("Please enter a player's first name.") #address error
        last_name = st.text_input('Enter a players last name:')
        last_name = last_name.strip()
        if " " in last_name:
            st.error("Please do not include whitespace in the input.") #address error
        if last_name.isspace():
            st.warning("Please enter a player's last name.") #address error

        if first_name and last_name:
            player_info = playerid_lookup(last_name, first_name) #lookup player id to input into the statcast_batter()
            pid = player_info['key_mlbam']
            st.markdown("#### Player ID")
            mlbid = pid.iloc[0] #get the player id value
            st.write(mlbid) #display the id
            name = first_name + " " + last_name #combine name for plot
           
            #provide acceptable entries for fields
            st.markdown("#### Here are the acceptable field entries:")
            fields = ['angels', 'astros', 'athletics', 'blue_jays', 'braves', 'brewers', 'cardinals', 'cubs', 'diamondbacks', 'dodgers', 'generic', 'giants', 'indians', 'mariners', 'marlins', 'mets', 'nationals', 'orioles', 'padres', 'phillies', 'pirates', 'rangers', 'rays', 'red_sox', 'reds', 'rockies', 'royals', 'tigers', 'twins', 'white_sox', 'yankees']
            fieldnames = pd.DataFrame(fields, columns = ['Fields']) 
            st.dataframe(fieldnames) #display fields
            stadium = st.text_input("Enter MLB team for stadium.") #user input for field name that data is overlayed on

            stadium = stadium.strip()
            if stadium.isspace():
                st.warning("Please enter a stadium.")
            if " " in stadium:
                st.error("Please do not include whitespace in the input.") #address error
            if stadium:
                data = statcast_batter('2022-04-07', '2022-10-02', mlbid) #pull data with player select in input based on id
                s = spraychart(data, stadium, title = name)
                fig = s.figure
                # Display the spraychart
                st.pyplot(fig)
                tot = data.events.value_counts() #display event totals
                st.dataframe(tot)

                #animation plots not implemented
                #fig2 = px.histogram(data, x ="events", color = "pitch_name", animation_frame = 'game_date', animation_group = 'events')
                #st.write(fig2)

                #fig3 = px.histogram(data, x ="events", color = "pitch_name")
                #st.write(fig3)
    #if the user causes an error let the user know to try new input            
    except IndexError as e:
        st.error("Incorrect Input. Please try another input.")
        
    #set up sidebar
    st.sidebar.markdown("# Explore player spray charts 📈")
    st.sidebar.markdown(" ")
    if st.sidebar.checkbox(" Select For Help ⚾️"): 
         st.sidebar.info("Steps to Follow: ")
         st.sidebar.info("1. Enter a hitter's first name and last name who played in the 2022 season. ")
         st.sidebar.info("2. Look to the acceptable field entries and select one to input in the stadium input box. ")
         st.sidebar.info("3. View player spray chart at the selected MLB Stadium and the frequency of each outcome. ")
         st.sidebar.markdown("### The drop down above ↑ includes different pages to navigate through. Select the next page to make predictions.")

#define prediction page       
def page3():
    st.title("Predict MLB At-Bat Outcomes ⚾️") 
    #Import necessary libraries
    from tensorflow import keras
    import matplotlib.pylab as plt
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import Dense, Activation, Dropout
    from tensorflow.keras.models import Sequential

    df = pd.read_csv("bsbc.csv") #read in the data
    #drop variables not needed and trim data
    df = df.drop(['Unnamed: 0', 'game_year'], axis = 1)
    df = df.drop(['player_name'], axis = 1)
    df = df[df["release_speed"] > 75]
    df = df[df['launch_speed'] > 55]
    df = df[(df["launch_angle"] < 50) & (df['launch_angle'] > 5)]
    df = df[df['bat_score'] <= 15]
    df = df[df['fld_score'] <= 15]
    df = df[(df["release_spin_rate"] < 3200) & (df['release_spin_rate'] > 1000)]
    #replace pitch name with number for easier user input
    df['pitch_name'] = df['pitch_name'].replace('4-Seam Fastball', 0)
    df['pitch_name'] = df['pitch_name'].replace('Slider', 1)
    df['pitch_name'] = df['pitch_name'].replace('Sinker', 2)
    df['pitch_name'] = df['pitch_name'].replace('Changeup', 3)
    df['pitch_name'] = df['pitch_name'].replace('Curveball', 4)
    #replace top and bot to 1 and 0
    df['inning_topbot'] = df['inning_topbot'].replace('Top', 1)
    df['inning_topbot'] = df['inning_topbot'].replace('Bot', 0)
    #if the baserunning variables are greater than zero there is someone on base
    df['on_1b'] = np.where(df['on_1b'] > 0, 1,0)
    df['on_2b'] = np.where(df['on_1b'] > 0, 1,0)
    df['on_3b'] = np.where(df['on_1b'] > 0, 1,0)
    #encode and seperate out outcome events
    from sklearn.preprocessing import OneHotEncoder
    df = pd.get_dummies(df, columns = ['events'])
    #get random sample of whole dataset
    df = df.sample(frac = 1)
    #define X
    X = df.drop(['game_type', 'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle', 
                 'events_double',
           'events_field_error', 'events_field_out', 'events_hit_by_pitch',
           'events_home_run', 'events_single', 'events_strikeout', 'events_triple',
           'events_walk', 'game_date'], 
                axis = 1)
    #define y
    y = df.drop(['release_speed', 'batter', 'pitcher', 'zone', 'balls', 'strikes',
           'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'inning_topbot',
           'launch_speed', 'launch_angle', 'effective_speed', 'release_spin_rate',
           'game_pk', 'pitch_name', 'bat_score', 'fld_score', 'delta_home_win_exp',
           'delta_run_exp', 'hp_to_1b', 'sprint_speed', 'game_date', 'game_type', 'estimated_ba_using_speedangle',
           'estimated_woba_using_speedangle'], axis = 1)


    #Split the data into 80% training and 20% testing
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    #Standardize the data
    from sklearn.preprocessing import StandardScaler
    s = StandardScaler()
    X_train = s.fit_transform(X_train)
    X_test = s.transform(X_test)

    #Building the artifical neural network
    model = Sequential()
    model.add(Dense(22, activation = 'relu', input_shape = (24,))) #24 input variables
    model.add(Dense(20, activation = 'relu'))
    model.add(Dense(9, activation = 'softmax')) #9 output possiblities
    #compile model and fit with 15 epochs
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs = 1, validation_split = 0.1, batch_size=10, verbose=1)

    pred = model.predict(X_test)
    
    #define outcome names to use with prediction
    names = ['Double', 'Field Error', 'Field Out', 'Hit by Pitch', 'Home Run', 'Single', 'Strikeout', 'Triple', 'Walk']
    

    #function to take in input data transform it and make prediction
    def predict(inputs):
        inputs = s.transform(inputs)
        prediction = model.predict(inputs)
        return prediction
    
     
    #user input for the model
    release_speed = st.number_input('Pitch Release Speed:', min_value=70, max_value=103, value = 88, help = 'Pitch velocity.')
    st.info("Keep in mind that data and information is coming from the 2022 season and roster changes are only reflected for player images.")
    
    hitter = st.selectbox("Please select a hitter.", ['Bo Bichette: TOR (SS/R)', 'Vladimir Guerrero Jr.: TOR (1B/R)', 'Anthony Santander: BAL (RF/S)', 'Randy Arozarena: TB (LF/R)', 'Xander Bogaerts: BOS (SS/R)', 'Aaron Judge: NYY (CF/R)', 'Andres Gimenez: CLE (2B/L)', 'Salvador Perez: KC (C/R)', 'Javier Baez: DET (SS/R)', 'Carlos Correa: MIN (SS/R)', 'Jose Abreu: CWS (1B/R)', 'Mike Trout: LAA (CF/R)', 'Yordan Alvarez: HOU (DH/L)', 'Julio Rodriguez: SEA (CF/R)', 'Nathaniel Lowe: TEX (1B/L)', 'Sean Murphy: OAK (C/R)', 'Dansby Swanson: ATL (SS/R)', 'Austin Riley: ATL (3B/R)', 'Garrett Cooper: MIA (1B/R)', 'Pete Alonso: NYM (1B/R)', 'Juan Soto: WSH (RF/L)','Rhys Hoskins: PHI (1B/R)', 'Paul Goldschmidt: STL (1B/R)', 'Willy Adames: MIL (SS/R)', 'Bryan Reynolds: PIT (CF/S)', 'Nico Hoerner: CHC (SS/R)','Kyle Farmer: CIN (SS/R)', 'Freddie Freeman: LAD (1B/L)', 'Christian Walker: ARI (1B/R)', 'Manny Machado: SD (3B/R)', 'Joc Pederson: SF (LF/L)', 'CJ Cron: COL (1B/R)'], help = 'Select an AL or NL Hitter.')
    if hitter == "Bo Bichette: TOR (SS/R)":
        hitter = 666182
        from PIL import Image 
        bo = Image.open('Images/bo.jpeg')
        st.image(bo)
    elif hitter == "Vladimir Guerrero Jr.: TOR (1B/R)":
        hitter = 665489
        from PIL import Image 
        vlad = Image.open('Images/vlad.jpeg')
        st.image(vlad)
    elif hitter == 'Anthony Santander: BAL (RF/S)':
        hitter = 623993
        from PIL import Image 
        AS = Image.open('Images/AS.jpeg')
        st.image(AS)
    elif hitter == 'Randy Arozarena: TB (LF/R)':
        hitter = 668227
        from PIL import Image 
        randy = Image.open('Images/randy.jpeg')
        st.image(randy)
    elif hitter == 'Xander Bogaerts: BOS (SS/R)':
        hitter = 593428
        from PIL import Image 
        xander = Image.open('Images/xander.jpeg')
        st.image(xander)
    elif hitter == 'Aaron Judge: NYY (CF/R)':
        hitter = 592450
        from PIL import Image 
        judge = Image.open('Images/judge.jpeg')
        st.image(judge)
    elif hitter == 'Andres Gimenez: CLE (2B/L)':
        hitter = 665926
        from PIL import Image 
        ag = Image.open('Images/ag.jpeg')
        st.image(ag)
    elif hitter == 'Salvador Perez: KC (C/R)':
        hitter = 521692
        from PIL import Image 
        salv = Image.open('Images/salv.jpeg')
        st.image(salv)
    elif hitter == 'Javier Baez: DET (SS/R)':
        hitter = 595879
        from PIL import Image 
        javi = Image.open('Images/javi.jpeg')
        st.image(javi)
    elif hitter == 'Carlos Correa: MIN (SS/R)':
        hitter = 621043
        from PIL import Image 
        correa = Image.open('Images/correa.jpeg')
        st.image(correa)
    elif hitter == 'Jose Abreu: CWS (1B/R)':
        hitter = 547989
        from PIL import Image 
        abreu = Image.open('Images/abreu.jpeg')
        st.image(abreu)
    elif hitter == 'Mike Trout: LAA (CF/R)':
        hitter = 545361
        from PIL import Image 
        trout = Image.open('Images/trout.jpeg')
        st.image(trout)
    elif hitter == 'Yordan Alvarez: HOU (DH/L)':
        hitter = 670541
        from PIL import Image 
        yordan = Image.open('Images/yordan.jpeg')
        st.image(yordan)
    elif hitter == 'Julio Rodriguez: SEA (CF/R)':
        hitter = 677594
        from PIL import Image 
        julio = Image.open('Images/julio.jpeg')
        st.image(julio)
    elif hitter == 'Nathaniel Lowe: TEX (1B/L)':
        hitter = 663993
        from PIL import Image 
        lowe = Image.open('Images/lowe.jpeg')
        st.image(lowe)
    elif hitter == 'Sean Murphy: OAK (C/R)':
        hitter = 669221
        from PIL import Image 
        murphy = Image.open('Images/murphy.jpeg')
        st.image(murphy)
    elif hitter == 'Dansby Swanson: ATL (SS/R)':
        hitter = 621020
        from PIL import Image 
        swanson = Image.open('Images/swanson.jpeg')
        st.image(swanson)
    elif hitter == 'Austin Riley: ATL (3B/R)':
        hitter = 663586
        from PIL import Image 
        riley = Image.open('Images/riley.jpeg')
        st.image(riley)
    elif hitter == 'Garrett Cooper: MIA (1B/R)':
        hitter = 643265
        from PIL import Image 
        coop = Image.open('Images/coop.jpeg')
        st.image(coop)
    elif hitter == 'Pete Alonso: NYM (1B/R)':
        hitter = 624413
        from PIL import Image 
        pete = Image.open('Images/pete.jpeg')
        st.image(pete)  
    elif hitter == 'Juan Soto: WSH (RF/L)':
        hitter = 665742
        from PIL import Image 
        soto = Image.open('Images/soto.jpeg')
        st.image(soto) 
    elif hitter == 'Rhys Hoskins: PHI (1B/R)':
        hitter = 656555
        from PIL import Image 
        rhys = Image.open('Images/rhys.jpeg')
        st.image(rhys)  
    elif hitter == 'Paul Goldschmidt: STL (1B/R)':
        hitter = 502671
        from PIL import Image 
        gold = Image.open('Images/gold.jpeg')
        st.image(gold)  
    elif hitter == 'Willy Adames: MIL (SS/R)':
        hitter = 642715
        from PIL import Image 
        willy = Image.open('Images/willy.jpeg')
        st.image(willy)  
    elif hitter == 'Bryan Reynolds: PIT (CF/S)':
        hitter = 668804
        from PIL import Image 
        bryan = Image.open('Images/bryan.jpeg')
        st.image(bryan)  
    elif hitter == 'Nico Hoerner: CHC (SS/R)':
        hitter = 663538
        from PIL import Image 
        nico = Image.open('Images/nico.jpeg')
        st.image(nico)  
    elif hitter == 'Kyle Farmer: CIN (SS/R)':
        hitter = 571657
        from PIL import Image 
        farmer = Image.open('Images/farmer.jpeg')
        st.image(farmer)  
    elif hitter == 'Freddy Freeman: LAD (1B/L)':
        hitter = 518692
        from PIL import Image 
        freddy = Image.open('Images/freddy.jpeg')
        st.image(freddy)  
    elif hitter == 'Christian Walker: ARI (1B/R)':
        hitter = 572233
        from PIL import Image 
        cwalker = Image.open('Images/cwalker.jpeg')
        st.image(cwalker)  
    elif hitter == 'Manny Machado: SD (3B/R)':
        hitter = 592518
        from PIL import Image 
        manny = Image.open('Images/manny.jpeg')
        st.image(manny)  
    elif hitter == 'Joc Pederson: SF (LF/L)':
        hitter = 592626
        from PIL import Image 
        joc = Image.open('Images/joc.jpeg')
        st.image(joc)  
    elif hitter == 'CJ Cron: COL (1B/R)':
        hitter = 543068
        from PIL import Image 
        cron = Image.open('Images/cron.jpeg')
        st.image(cron)  
   
    pitcheral = st.selectbox("Please select an AL pitcher.", ['Tyler Wells: BAL (R)', 'Drew Rasmussen: TB (R)', 'Nick Pivetta: BOS (R)', 'Nestor Cortes: NYY (L)', 'Triston McKenzie: CLE (R)', 'Brady Singer: KC (R)', 'Tarik Skubal: DET (L)', 'Sonny Gray: MIN (R)', 'Dylan Cease: CWS (R)', 'Shohei Ohtani: LAA (R)', 'Justin Verlander: HOU (R)', 'James Kaprielian: OAK (R)', 'Chris Flexen: SEA (R)', 'Martin Perez: TEX (L)', 'Max Fried: ATL (L)', 'Sandy Alcantara: MIA (R)', 'Max Scherzer: NYM (R)', 'Aaron Nola: PHI (R)', 'Paolo Espino: WSH (R)', 'Corbin Burnes: MIL (R)', 'Marcus Stroman: CHC (R)', 'Nick Lodolo: CIN (L)', 'Miles Mikolas: STL (R)', 'Jose Quintana: PIT (L)', 'Tony Gonsolin: LAD (R)', 'Zac Gallen: ARI (R)', 'Yu Darvish: SD (R)','Carlos Rodon: SF (L)', 'Ryan Feltner: COL (R)'], help = 'Select an American or National League Pitcher.')
    if pitcheral == 'Tyler Wells: BAL (R)':
        pitcheral = 669330
        wells = Image.open('Images/wells.jpeg')
        st.image(wells)
    elif pitcheral == 'Drew Rasmussen: TB (R)':
        pitcheral = 656876
        ras = Image.open('Images/ras.jpeg')
        st.image(ras)
    elif pitcheral == 'Nick Pivetta: BOS (R)':
        pitcheral = 601713
        piv = Image.open('Images/piv.jpeg')
        st.image(piv)
    elif pitcheral == 'Nestor Cortes: NYY (L)':
        pitcheral = 6641482
        from PIL import Image 
        nestor = Image.open('Images/nestor.jpeg')
        st.image(nestor)
    elif pitcheral == 'Triston McKenzie: CLE (R)':
        pitcheral = 663474
        tmck = Image.open('Images/tmck.jpeg')
        st.image(tmck)
    elif pitcheral == 'Brady Singer: KC (R)':
        pitcheral = 6641482
        singer = Image.open('Images/singer.jpeg')
        st.image(singer)
    elif pitcheral == 'Tarik Skubal: DET (L)':
        pitcheral = 669373
        tarik = Image.open('Images/tarik.jpeg')
        st.image(tarik)
    elif pitcheral ==  'Sonny Gray: MIN (R)':
        pitcheral = 543243
        gray = Image.open('Images/gray.jpeg')
        st.image(gray)
    elif pitcheral ==  'Dylan Cease: CWS (R)':
        pitcheral = 656302
        cease = Image.open('Images/cease.jpeg')
        st.image(cease)
    elif pitcheral ==  'Shohei Ohtani: LAA (R)':
        pitcheral = 660271
        sho = Image.open('Images/sho.jpeg')
        st.image(sho)
    elif pitcheral ==  'Justin Verlander: HOU (R)':
        pitcheral = 434378
        ver = Image.open('Images/ver.jpeg')
        st.image(ver)
    elif pitcheral ==  'James Kaprielian: OAK (R)':
        pitcheral = 621076
        kap = Image.open('Images/kap.jpeg')
        st.image(kap)
    elif pitcheral ==  'Chris Flexen: SEA (R)':
        pitcheral = 623167
        flexen = Image.open('Images/flexen.jpeg')
        st.image(flexen)
    elif pitcheral ==  'Martin Perez: TEX (L)':
        pitcheral = 527048
        mperez = Image.open('Images/mperez.jpeg')
        st.image(mperez)
    elif pitcheral == 'Max Fried: ATL (L)':
        pitcheral = 608331
        fried= Image.open('Images/fried.jpeg')
        st.image(fried) 
    elif pitcheral == 'Sandy Alcantara: MIA (R)':
        pitcheral = 645261
        sandy = Image.open('Images/sandy.jpeg')
        st.image(sandy) 
    elif pitcheral == 'Max Scherzer: NYM (R)':
        pitcheral = 453286
        maxs = Image.open('Images/maxs.jpeg')
        st.image(maxs)
    elif pitcheral == 'Aaron Nola: PHI (R)':
        pitcheral = 605400
        nola = Image.open('Images/nola.jpeg')
        st.image(nola)
    elif pitcheral == 'Paolo Espino: WSH (R)':
        pitcheral = 502179
        espino = Image.open('Images/espino.jpeg')
        st.image(espino)
    elif pitcheral == 'Corbin Burnes: MIL (R)':
        pitcheral = 669203
        burnes = Image.open('Images/burnes.jpeg')
        st.image(burnes)
    elif pitcheral == 'Marcus Stroman: CHC (R)':
        pitcheral = 573186
        stroman = Image.open('Images/stroman.jpeg')
        st.image(stroman)
    elif pitcheral == 'Nick Lodolo: CIN (L)':
        pitcheral = 666157
        lodolo = Image.open('Images/lodolo.jpeg')
        st.image(lodolo)
    elif pitcheral == 'Miles Mikolas: STL (R)':
        pitcheral = 571945
        miles = Image.open('Images/miles.jpeg')
        st.image(miles)
    elif pitcheral == 'Jose Quintana: PIT (L)':
        pitcheral = 500779
        quin = Image.open('Images/quin.jpeg')
        st.image(quin)
    elif pitcheral == 'Tony Gonsolin: LAD (R)':
        pitcheral = 664062
        gons = Image.open('Images/gons.jpeg')
        st.image(gons)
    elif pitcheral == 'Zac Gallen: ARI (R)':
        pitcheral = 668678
        gallen = Image.open('Images/gallen.jpeg')
        st.image(gallen)
    elif pitcheral == 'Yu Darvish: SD (R)':
        pitcheral = 506433
        yu = Image.open('Images/yu.jpeg')
        st.image(yu)
    elif pitcheral == 'Carlos Rodon: SF (L)':
        pitcheral = 607074
        rodon = Image.open('Images/rodon.jpeg')
        st.image(rodon)
    elif pitcheral == 'Ryan Feltner: COL (R)':
        pitcheral = 663372
        felt = Image.open('Images/felt.jpeg')
        st.image(felt)
        

    #else:
        #pitchernl = st.selectbox("Please select an NL pitcher.", ['Max Fried: ATL (L)', 'Sandy Alcantara: MIA (R)', 'Max Scherzer: NYM (R)', 'Paolo Espino: WSH (R)', 'Aaron Nola: PHI (R)', 'Corbin Burnes: MIL (R)', 'Miles Mikolas: STL (R)', 'Marcus Stroman: CHC (R)', 'Jose Quintana: PIT (L)', 'Nick Lodolo: CIN (L)', 'Zac Gallen: ARI (R)', 'Clayton Kershaw: LAD (L)', 'Carlos Rodon: SF (L)', 'Blake Snell: SD (L)', 'Ryan Feltner: COL (R)'])
    
    #display a strikezone image to help with user input selection
    showzone = st.checkbox('Display Strike Zone')
    if showzone:
        from PIL import Image 
        zonepng = Image.open('Images/zone.png')
        st.image(zonepng)
    zone = st.number_input('Strike Zone Location: ', min_value=1, max_value=14, value = 5, help = "Select a strike zone location - location of the ball when it crosses the plate from the catcher's perspective.")

    #select more options
    with st.expander('Select More Options'): 
        balls = st.number_input('Number of Balls: ', min_value=0, max_value=3)

        strikes = st.number_input('Number of Strikes:', min_value=0, max_value=2)


        st.markdown("##### Baserunners")
        on_1b = st.checkbox('Runner on 1B')
        if on_1b:
            on_1b = 1
        else:
            on_1b = 0
        on_2b = st.checkbox('Runner on 2B')
        if on_2b:
            on_2b = 1
        else:
            on_2b = 0
        on_3b = st.checkbox('Runner on 3B')
        if on_3b:
            on_3b = 1
        else:
            on_3b = 0

        outs_when_up = st.radio('Outs When Up: ', ('0','1', '2'), help = 'The number of outs during the at-bat.')
        if outs_when_up == '0':
            outs_when_up = 0
        elif outs_when_up == '1':
            outs_when_up = 1
        elif outs_when_up == '2':
            outs_when_up = 2
            
        inning = st.number_input('Inning: ', min_value=1, max_value=10, value=1, help = 'Inning during at-bat.')

        inning_topbot = st.radio('Top or Bottom of Inning: ', ('Top', 'Bottom'), help = 'Top or bottom of the inning.')
        if inning_topbot == 'Top':
            inning_topbot = 1
        else:
            inning_topbot = 0

        launch_speed = st.number_input('Exit Velocity: ', min_value=60, max_value=114, value = 82, help = 'Batter Exit Velocity')

        launch_angle = st.number_input('Launch Angle: ', min_value=5, max_value=50, value = 16)

        effective_speed = st.number_input('Effective Speed: ', min_value=70, max_value=103, value = 88, help = 'Speed based on extension of pitcher release')

        release_spin_rate = st.number_input('Release Spin Rate: ', min_value=1000, max_value=3200, value = 2200, help = 'Spin rate of pitch tracked by Statcast')

        #game_pk = st.number_input('MLB Stadium: ', min_value=663419, max_value=663419, help = 'Ballpark game is being played in.')
        game_pk = st.selectbox('MLB Stadium: ', ['CLE', 'CWS', 'CIN', 'MIL', 'BAL', 'MIA', 'LAA', 'TB', 'BOS', 'ATL', 'KC'])
        if game_pk == 'CLE':
            game_pk = 662958
        elif game_pk == 'CWS':
            game_pk = 661481
        elif game_pk == 'CIN':
            game_pk = 662995
        elif game_pk == 'MIL':
            game_pk = 661193
        elif game_pk == 'BAL':
            game_pk = 661279
        elif game_pk == 'MIA':
            game_pk = 661355
        elif game_pk == 'LAA':
            game_pk = 663442
        elif game_pk == 'TB':
            game_pk = 661944
        elif game_pk == 'BOS':
            game_pk = 663222
        elif game_pk == 'ATL':
            game_pk = 661552
        elif game_pk == 'KC':
            game_pk = 662638
                               

        pitch_name = st.selectbox('Pitch Name: ', ['4-Seam Fastball', 'Slider', 'Sinker', 'Changeup', 'Curveball'], help = "The name of pitch thrown.")
        if  pitch_name == '4-Seam Fastball':
            pitch_name = 0
        elif  pitch_name == 'Slider':
            pitch_name = 1 
        elif  pitch_name == 'Sinker':
            pitch_name = 2
        elif  pitch_name == 'Changeup':
            pitch_name = 3
        elif  pitch_name == 'Curveball':
            pitch_name = 4

        bat_score = st.number_input('Hitting Team Score: ', min_value=0, max_value=15, help = 'The number of runs the hitting team has.')

        fld_score = st.number_input('Fielding Team Score: ', min_value=0, max_value=15, help = 'The number of runs the fielding team has.')

        win_exp = st.number_input('Win Exp: ', min_value=0.1, max_value=0.25, help = 'Win Expectancy')
    
        run_exp = st.number_input('Run Exp: ', min_value = 1, max_value=3, help = 'Run Expectancy')
  
        hm1 = st.slider("Home to First Time:  ", 4.0, 5.0, help = "Batter's time it takes to run to first base.")

        speed = st.slider("Sprint Speed: ", 23.0, 30.7, help = 'How fast the batter can run.')

    #user inputs
    inputs = pd.DataFrame([[release_speed, hitter, pitcheral, zone, balls, strikes, on_3b, on_2b, on_1b, outs_when_up, inning, inning_topbot, launch_speed, launch_angle, effective_speed, release_spin_rate, game_pk, pitch_name, bat_score, fld_score, win_exp, run_exp, hm1, speed]], columns = ['release_speed', 'hitter', 'pitcheral', 'zone', 'balls', 'strikes', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'inning_topbot', 'launch_speed', 'launch_angle', 'effective_speed', 'release_spin_rate', 'game_pk', 'pitch_name', 'bat_score', 'fld_score',' win_exp', 'run_exp', 'hm1', 'speed'])
    #make predictions
    if st.button('Predict At-Bat'):
        Abpredict = predict(inputs)
        st.success(names[np.argmax(Abpredict)]) #display prediction
        
        #probability that outcome will occur
        def sci(num):
            return '{:.2f}'.format(num)
        predictions = np.vectorize(sci)(Abpredict)
        st.write(predictions)
    #sidebar setup
    st.sidebar.markdown("# Make At-Bat Predictions ⚾️🔍")
    st.sidebar.markdown(" ")
    if st.sidebar.checkbox(" Select For Help ⚾️"): 
         st.sidebar.info("Steps to make a prediction: ")
         st.sidebar.info("1. Enter pitch release speed")
         st.sidebar.info("2. Select a hitter")
         st.sidebar.info("3. Select a pitcher")
         st.sidebar.info("4. Select a strike zone location")
         st.sidebar.info("5. Expand for more options and select other input if desired")
         st.sidebar.info("6. Click the 'Predict At-Bat' Button to display predicted outcome")
#dict for pages
page_names_to_funcs = {
    "Welcome Page": main_page,
    "Data Exploration and Visualization": page2,
    "MLB At-Bat Predictor": page3 
    }
#select pages
selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
