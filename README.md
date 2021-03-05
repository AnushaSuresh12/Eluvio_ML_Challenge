# Eluvio_ML_Challenge
The repository consists of an Ensemble model to predict the scene segmentation for each movie with given set of features (place,cast,action,audio).

In order to run the model, we need to provide the dataset link in the filename = "directory of the dataset" in the ScenePrediction.py

To generate the output like the format given below you need to give this command

python3 ScenePrediciton.py

Sample output looks like:

Scores: {
    "AP": 0.9929857410829752,
    "mAP": 0.9926434843640202,
    "Miou": 0.9507961996655592
}



