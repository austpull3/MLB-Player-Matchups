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

    df = pd.read_csv("bsbc.csv")
    
    df = df.drop(['Unnamed: 0', 'game_year'], axis = 1)
    df = df.drop(['player_name'], axis = 1)
    df = df[df["release_speed"] > 75]
    df = df[df['launch_speed'] > 55]
    df = df[(df["launch_angle"] < 50) & (df['launch_angle'] > 5)]
    df = df[df['bat_score'] <= 15]
    df = df[df['fld_score'] <= 15]
    df = df[(df["release_spin_rate"] < 3200) & (df['release_spin_rate'] > 1000)]

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
    model.add(Dense(22, activation = 'relu', input_shape = (24,)))
    model.add(Dense(20, activation = 'relu'))
    model.add(Dense(9, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs = 15, validation_split = 0.1, batch_size=10, verbose=1)

    pred = model.predict(X_test)
    
    model.save('model.h5')
