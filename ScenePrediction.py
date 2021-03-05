import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from evalateSceneSeg import *

if __name__ == '__main__':
    ground_truth = {}
    prediction = {}
    shot_end_frame = {}
    scores = {}
    filename = 'Data/Movie_Data'
    output_location = 'Result/'
    for file in os.listdir('Data/Movie_Data'):
        x = pickle.load(open(os.path.join(filename,file), "rb"))
        ground_truth[x["imdb_id"]] = x["scene_transition_boundary_ground_truth"].numpy()
        prediction[x["imdb_id"]] = x["scene_transition_boundary_prediction"].numpy()
        shot_end_frame[x["imdb_id"]] = x["shot_end_frame"].numpy()
        # Now we need to generate Input dataframe
        df1 = pd.DataFrame(x["place"]).astype("float")
        df2 = pd.DataFrame(x["cast"]).astype("float")
        df3 = pd.DataFrame(x["action"]).astype("float")
        df4 = pd.DataFrame(x["audio"]).astype("float")
        df5 = pd.DataFrame(x["scene_transition_boundary_ground_truth"]).astype("float")
        df = pd.concat([df1,df2,df3,df4,df5],axis=1)
        df = df.head(-1)
        X = df.iloc[ : , :-1].values
        Y = df.iloc[: , -1].values
        random_forest = RandomForestRegressor(n_estimators=10,random_state=42)
        random_forest.fit(X,Y)
        Y_pred = random_forest.predict(X)
        Y_pred = pd.DataFrame(Y_pred)
        prediction[x["imdb_id"]] = Y_pred
        Y_pred.to_csv(os.path.join(output_location,(file+".csv")),sep=',',index = False)
        scores["AP"], scores["mAP"], _ = calc_ap(ground_truth, prediction)
        scores["Miou"], _ = calc_miou(ground_truth, prediction, shot_end_frame)
        print("Scores:", json.dumps(scores, indent=4))









