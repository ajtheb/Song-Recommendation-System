
import pandas as pd
import numpy as np
import spotipy 
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def feature_extraction(sp,playlist_link):
    ## feature extraction
    playlist_URI = playlist_link.split("/")[-1].split("?")[0]
    track_uri=[]
    track_name=[]
    artists=[]
    allbum_type=[]
    album_name=[]
    track_pop=[]
    audio_features=[]
    track_genres=[]
    artist_name=[]
    for track in sp.playlist_tracks(playlist_URI)["items"]:
        #URI
        track_uri.append(track["track"]["uri"])
        
        #Track name
        track_name.append(track["track"]["name"])

        
        artists.append(track["track"]["artists"])

        #Genre of the most followed artist of the track 
        genres= []
        for x in track["track"]["artists"]:
            genres.append(( sp.artist(x['uri'])['genres'],sp.artist(x['uri'])['followers']['total'],sp.artist(x['uri'])['name']))
        genres.sort(key=lambda x: x[1],reverse=True)
        track_genres.append(",".join(genres[0][0]))
        artist_name.append(genres[0][2])

        #Album
        album_name.append(track["track"]["album"]["name"])
        
        #Popularity of the track
        track_pop.append(track["track"]["popularity"])

        audio_features.append(sp.audio_features(track['track']['uri']))
    
    danceability = []
    energy = []
    key = []
    loudness = []
    speechiness = []
    acousticness = []
    instrumentalness = []
    liveness = []
    valence = []
    tempo = []
    duration_ms = []
    for i in audio_features:
        danceability.append(i[0]['danceability'])
        energy.append(i[0]['energy'])
        key.append(i[0]['key'])
        loudness.append(i[0]['loudness'])
        speechiness.append(i[0]['speechiness'])
        acousticness.append(i[0]['acousticness'])
        instrumentalness.append(i[0]['instrumentalness'])
        liveness.append(i[0]['liveness'])
        valence.append(i[0]['valence'])
        tempo.append(i[0]['tempo'])
        duration_ms.append(i[0]['duration_ms'])
    df=pd.DataFrame({"Track_uri":track_uri,
                    "Album_name":album_name,
                    "Track_name":track_name,
                    "Track_genres":track_genres,
                    "Artist_name":artist_name,
                    "Track_popularity":track_pop,
                    "danceability":danceability,
                    "energy":energy,"key":key,
                    "loudness":loudness,
                    "speechiness":speechiness,
                    "acousticness":acousticness,
                    "instrumentalness":instrumentalness,
                    "liveness":liveness,
                    "valence":valence,
                    "tempo":tempo,
                    "duration_ms":duration_ms})
    return df

def create_feature_set(df):

    feature_cols=['acousticness', 'danceability', 'duration_ms', 'energy',
                    'instrumentalness', 'key', 'liveness', 'loudness',
                    'speechiness', 'tempo', 'valence',]

    scaler = MinMaxScaler()
    normalized_df =pd.DataFrame(scaler.fit_transform(df[feature_cols]),columns=feature_cols)
        
    df['genres_list'] = df['Track_genres'].apply(lambda x: x.split(","))
    # TF-IDF implementation
    tfidf = TfidfVectorizer()
    tfidf_matrix =  tfidf.fit_transform(df['genres_list'].apply(lambda x: " ".join(x)))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names()]
    #genre_df.drop(columns='genre|unknown')
    #genre_df.reset_index(drop = True, inplace=True)
    final = pd.concat([genre_df, normalized_df], axis = 1)
    final['Track_uri']=df['Track_uri'].values
    return final

def main(your_playlist):
    cid=''
    secret=''
    #Authentication - without user
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

    bollywood_songs = 'https://open.spotify.com/playlist/7sTkp2X5Aq84v9w9UtfkaF?si=1499ad823a874dbf'
    df=feature_extraction(sp,your_playlist)
    all_df=feature_extraction(sp,bollywood_songs)
    DF=pd.concat([df,all_df],axis=0)
    DF=DF.drop_duplicates('Track_name')

    DF_feature_set=create_feature_set(DF)
    playlist_feature_set=DF_feature_set[DF_feature_set['Track_uri'].isin(df['Track_uri'].values)]
    non_playlist_feature_set=DF_feature_set[~(DF_feature_set['Track_uri'].isin(df['Track_uri'].values))]

    non_play=DF[~DF['Track_uri'].isin(df['Track_uri'].values)]
    non_playlist_feature_set.drop(['Track_uri'],axis=1,inplace=True)
    playlist_feature_set.drop(['Track_uri'],axis=1,inplace=True)
    playlist_feature_set_1 = playlist_feature_set.sum(axis = 0)
    
    # Find cosine similarity between the playlist and the complete song set
    non_play['sim'] = cosine_similarity(non_playlist_feature_set, playlist_feature_set_1.values.reshape(1, -1))[:,0]
    non_playlist_df_top_40 = non_play.sort_values('sim',ascending = False).head(40)
    print(non_playlist_df_top_40.head(10))
    return non_playlist_df_top_40
