import streamlit as st
import plotly.express as px
from pybaseball import playerid_lookup
from pybaseball import statcast_batter, spraychart



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
    
    from PIL import Image 
    image7 = Image.open('Images/dansby.png')
    st.image(image7)

    
    
    
    

def page2():
    
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

df = pd.read_csv("/Users/austinpullar/Desktop/bsbc.csv")

print(df.head())
print(df.shape)
df = df.drop(['Unnamed: 0', 'game_year'], axis = 1)
print(df.shape)
print(df.info())
print(df.isna().sum())
df = df.drop(['player_name'], axis = 1)

df = df[df["release_speed"] > 75]
df = df[df['launch_speed'] > 55]
df = df[(df["launch_angle"] < 50) & (df['launch_angle'] > 5)]
df = df[df['bat_score'] <= 15]
df = df[df['fld_score'] <= 15]
df = df[(df["release_spin_rate"] < 3200) & (df['release_spin_rate'] > 1000)]

#from sklearn.preprocessing import LabelEncoder
#le=LabelEncoder()
#df["pitch_name"]=le.fit_transform(df["pitch_name"])

df['pitch_name'] = df['pitch_name'].replace('4-Seam Fastball', 0)
df['pitch_name'] = df['pitch_name'].replace('Slider', 1)
df['pitch_name'] = df['pitch_name'].replace('Sinker', 2)
df['pitch_name'] = df['pitch_name'].replace('Changeup', 3)
df['pitch_name'] = df['pitch_name'].replace('Curveball', 4)

df['inning_topbot'] = df['inning_topbot'].replace('Top', 1)
df['inning_topbot'] = df['inning_topbot'].replace('Bot', 0)


df['on_1b'] = np.where(df['on_1b'] > 0, 1,0)
df['on_2b'] = np.where(df['on_1b'] > 0, 1,0)
df['on_3b'] = np.where(df['on_1b'] > 0, 1,0)



from sklearn.preprocessing import OneHotEncoder
df = pd.get_dummies(df, columns = ['events'])

df = df.sample(frac = 1)

X = df.drop(['game_type', 'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle', 
             'events_double',
       'events_field_error', 'events_field_out', 'events_hit_by_pitch',
       'events_home_run', 'events_single', 'events_strikeout', 'events_triple',
       'events_walk', 'game_date'], 
            axis = 1)

y = df.drop(['release_speed', 'batter', 'pitcher', 'zone', 'balls', 'strikes',
       'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'inning_topbot',
       'launch_speed', 'launch_angle', 'effective_speed', 'release_spin_rate',
       'game_pk', 'pitch_name', 'bat_score', 'fld_score', 'delta_home_win_exp',
       'delta_run_exp', 'hp_to_1b', 'sprint_speed', 'game_date', 'game_type', 'estimated_ba_using_speedangle',
       'estimated_woba_using_speedangle'], axis = 1)

#print(y.columns)


#Split the data into 80% training and 20% testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


#Standardize the data
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)
#print(X_train)

#Building the artifical neural network

model = Sequential()
model.add(Dense(22, activation = 'relu', input_shape = (24,)))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(9, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs = 15, validation_split = 0.1, batch_size=10, verbose=1)

#print(model.evaluate(X_test, y_test))

pred = model.predict(X_test)


#print(X_test[3])

test1 = model.predict([[0.88821384, -0.69551737,  0.84662357, -1.38522099, -0.29117606,  0.84870214
, -0.29530991, -0.47319155, -0.64135969, -1.19864502, -0.35934049, -1.02109916
  ,0.28160851 , 0.59447611 , 0.91606357, -0.41449101 , 0.25445557, -1.04564967
 ,-0.45465305  ,2.32810885, -0.10040018 ,-0.53267099, -0.22134649 , 0.29096857]])
#print(test1)

names = ['Double', 'Field Error', 'Field Out', 'Hit by Pitch', 'Home Run', 'Single', 'Strikeout', 'Triple', 'Walk']

#print(names[np.argmax(test1)])

c = np.argmax(pred, axis = 1)
#print(c[0:210])

'''
def predict(release_speed, batter, pitcher, zone, balls, strikes, on_3b, on_2b, on_1b, outs_when_up, inning, inning_topbot, launch_speed, launch_angle, effective_speed, release_spin_rate, game_pk, pitch_name, bat_score, fld_score, win_exp, run_exp, hm1, speed):
    prediction = model.predict(pd.DataFrame([[release_speed, batter, pitcher, zone, balls, strikes, on_3b, on_2b, on_1b, outs_when_up, inning, inning_topbot, launch_speed, launch_angle, effective_speed, release_spin_rate, game_pk, pitch_name, bat_score, fld_score, win_exp, run_exp, hm1, speed]], columns = [release_speed, batter, pitcher, zone, balls, strikes, on_3b, on_2b, on_1b, outs_when_up, inning, inning_topbot, launch_speed, launch_angle, effective_speed, release_spin_rate, game_pk, pitch_name, bat_score, fld_score, win_exp, run_exp, hm1, speed]))
    return prediction
'''
def predict(inputs):
    inputs = s.transform(inputs)
    st.write(inputs)
    print(inputs)
    un = s.inverse_transform(inputs)
    st.write(un)
    prediction = model.predict(inputs)
    return prediction



st.markdown(("### Release Speed"))
release_speed = st.number_input('Release speed', min_value=70, max_value=103)

st.markdown("## Select Batter:")
batter = st.selectbox('', ['Bryce Harper'])
if  batter == 'Bryce Harper':
    batter = 547180
    
st.markdown("## Select Pitcher:") 
pitcher = st.selectbox('', ['Kyle Wright'])
if  pitcher == 'Kyle Wright':
    pitcher = 657140
    
#st.markdown("### Zone 5 is the middle of the strike zone. 1, 3, 7, and 9 are the corners. ")
st.markdown("## Zone:")
zone = st.number_input(' ', min_value=1, max_value=14, help = "Select a strike zone location - location of the ball when it crosses the plate from the catcher's perspective.")

with st.expander('Select more options'): 
    st.markdown("## Number of Balls:")
    balls = st.number_input('  ', min_value=0, max_value=3)
    
    st.markdown("## Number of Strikes:")
    strikes = st.number_input('   ', min_value=0, max_value=2)


    st.markdown("### Baserunners")
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
    
    st.markdown("## Outs When Up:")
    outs_when_up = st.radio('Outs: ', ('0','1', '2'))
    if outs_when_up == '0':
        outs_when_up = 0
    elif outs_when_up == '1':
        outs_when_up = 1
    elif outs_when_up == '2':
        outs_when_up = 2
    #outs_when_up = st.number_input('    ', min_value=0, max_value=2, value=1)
    
    st.markdown("## Inning:")
    inning = st.number_input('     ', min_value=1, max_value=10, value=1)
    
    inning_topbot = st.radio('Top or Bottom of Inning', ('Top', 'Bottom'))
    if inning_topbot == 'Top':
        inning_topbot = 1
    else:
        inning_topbot = 0
        
    
    st.markdown("## Launch Speed:")
    launch_speed = st.number_input('      ', min_value=60, max_value=114)
    
    st.markdown("## Launch Angle:")
    launch_angle = st.number_input('       ', min_value=5, max_value=50)
    
    st.markdown("## Effective Speed:")
    effective_speed = st.number_input('          ', min_value=70, max_value=103)
    
    
    st.markdown("## Release Spin Rate:")
    release_spin_rate = st.number_input('         ', min_value=1000, max_value=3200)
    
    st.markdown("## MLB Stadium:")
    game_pk = st.number_input('           ', min_value=663419, max_value=663419)
    
    st.markdown("### Pitch name:")  
    pitch_name = st.selectbox('', ['4-Seam Fastball', 'Slider', 'Sinker', 'Changeup', 'Curveball'])
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
        
    bat_score = st.number_input('Hitting Team Score:', min_value=0, max_value=15)
    
    fld_score = st.number_input('Fielding Team Score:', min_value=0, max_value=15)
    
    win_exp = st.number_input('Win Exp:', min_value=-0.25, max_value=0.25)
    
    run_exp = st.number_input('Run Exp:', min_value=0, max_value=3)
    
    hm1 = st.slider("Home to First Time: ", 4.0, 5.0)
    
    speed = st.slider("Sprint Speed: ", 23.0, 30.7)


#inputs = pd.DataFrame([[release_speed, batter, pitcher, zone, balls, strikes, on_3b, on_2b, on_1b, outs_when_up, inning, inning_topbot, launch_speed, launch_angle, effective_speed, release_spin_rate, game_pk, pitch_name, bat_score, fld_score, win_exp, run_exp, hm1, speed]], columns = ['release_speed', 'batter', 'pitcher', 'zone', 'balls', 'strikes', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'inning_topbot', 'launch_speed', 'launch_angle', 'effective_speed', 'release_spin_rate', 'game_pk', 'pitch_name', 'bat_score', 'fld_score',' win_exp', 'run_exp', 'hm1', 'speed'])
inputs = pd.DataFrame([[release_speed, batter, pitcher, zone, balls, strikes, on_3b, on_2b, on_1b, outs_when_up, inning, inning_topbot, launch_speed, launch_angle, effective_speed, release_spin_rate, game_pk, pitch_name, bat_score, fld_score, win_exp, run_exp, hm1, speed]], columns = ['release_speed', 'batter', 'pitcher', 'zone', 'balls', 'strikes', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'inning_topbot', 'launch_speed', 'launch_angle', 'effective_speed', 'release_spin_rate', 'game_pk', 'pitch_name', 'bat_score', 'fld_score',' win_exp', 'run_exp', 'hm1', 'speed'])
print(inputs)
st.write(inputs)
prediction1 = predict(inputs)
st.write(names[np.argmax(prediction1)])


st.dataframe(inputs)



'''
if st.button('Predict At-Bat'):
    price = predict(release_speed, batter, pitcher, zone, balls, strikes, on_3b, on_2b, on_1b, outs_when_up, inning, inning_topbot, launch_speed, launch_angle, effective_speed, release_spin_rate, game_pk, pitch_name, bat_score, fld_score, win_exp, run_exp, hm1, speed)
    st.success(names[np.argmax(price)])
'''

    '''
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
    '''
    
    league = st.radio("Select National League or American League:", ("NL", "AL"))
    if league == "NL":
        col4, col5, col6 = st.columns(3)
        with col4:
            st.write("You selected the National League.")
            nlteam = st.radio("Select a National League Team.", ('Braves', 'Marlins', 'Mets', 'Nationals', 'Phillies', 'Brewers', 'Cardinals', 'Cubs', 'Pirates', 'Reds', 'D-backs', 'Dodgers', 'Giants', 'Padres', 'Rockies'))
            if nlteam == "Braves":
                with col5:
                    from PIL import Image 
                    braves = Image.open('Images/braves logo.png')
                    st.image(braves)
            elif nlteam == "Marlins":
                with col5:
                    from PIL import Image 
                    marlins = Image.open('Images/marlins.png')
                    st.image(marlins)
            elif nlteam == "Mets":
                with col5:
                    from PIL import Image 
                    mets = Image.open('Images/mets.png')
                    st.image(mets)
            elif nlteam == "Nationals":
                with col5:
                    from PIL import Image 
                    nationals = Image.open('Images/nationals.png')
                    st.image(nationals)
            elif nlteam == "Phillies": 
                with col5:
                    from PIL import Image 
                    phillies = Image.open('Images/phillies.jpeg')
                    st.image(phillies)
                    
            elif nlteam == "Brewers":
                with col5:
                    from PIL import Image 
                    brewers = Image.open('Images/brewers.png')
                    st.image(brewers)
                    
            elif nlteam == "Cardinals":
                with col5:
                    from PIL import Image 
                    cards = Image.open('Images/cardinals.png')
                    st.image(cards)
            elif nlteam == "Cubs":
                with col5:
                    from PIL import Image 
                    cubs = Image.open('Images/cubs.png')
                    st.image(cubs)
            elif nlteam == "Pirates":
                with col5:
                    from PIL import Image 
                    pit = Image.open('Images/pit.png')
                    st.image(pit)
            elif nlteam == "Reds":
                with col5:
                    from PIL import Image 
                    reds = Image.open('Images/reds.webp')
                    st.image(reds)
            elif nlteam == "D-backs":
                with col5:
                    from PIL import Image 
                    dbacks = Image.open('Images/dbacks.png')
                    st.image(dbacks)
            elif nlteam == "Dodgers":
                with col5:
                    from PIL import Image 
                    lad = Image.open('Images/laa.png')
                    st.image(lad)
            elif nlteam == "Dodgers":
                with col5:
                    from PIL import Image 
                    lad = Image.open('Images/laa.png')
                    st.image(lad)
            elif nlteam == "Giants":
                with col5:
                    st.markdown(" ")
                    from PIL import Image 
                    g = Image.open('Images/giants.jpeg')
                    st.image(g)
            elif nlteam == "Padres":
                with col5:
                    from PIL import Image 
                    sd = Image.open('Images/sd.png')
                    st.image(sd)
            elif nlteam == "Rockies":
                with col5:
                    from PIL import Image 
                    col = Image.open('Images/col.png')
                    st.image(col)
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
        '''
        player_name = st.text_input('Enter last name of player:')

        if player_name:
            player_info = playerid_lookup(player_name)
            st.write(player_info)
            plot = st.text_input("Enter player's key_mlbam:")
            if plot:
                data = statcast_batter('2022-04-07', '2022-10-02', plot)
                s = spraychart(data, 'generic', title = player_name)
                fig = s.figure
                # Display the spraychart
                st.pyplot(fig)
                tot = data.events.value_counts()
                st.write(tot)
                fig2 = px.bar(data, x = 'events', y = "pitch_type", animation_frame = "game_date", animation_group = "events")
                st.write(fig2)
        '''
        first_name = st.text_input('Enter a players first name:')
        last_name = st.text_input('Enter a players last name:')

        if first_name and last_name:
            player_info = playerid_lookup(last_name, first_name)
            st.write(player_info)
            name = first_name + " " + last_name
            plot = st.text_input("Enter player's key_mlbam:")
            stadium = st.text_input("Enter MLB team for stadium.")                               
            if plot:
                data = statcast_batter('2022-04-07', '2022-10-02', plot)
                if stadium:
                    s = spraychart(data, stadium, title = name)
                    fig = s.figure
                    # Display the spraychart
                    st.pyplot(fig)
                    tot = data.events.value_counts()
                    st.write(tot)
                    fig2 = px.bar(data, x = 'events', y = "pitch_type", animation_frame = "game_date", animation_group = "events")
                    st.write(fig2)
           
        
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
