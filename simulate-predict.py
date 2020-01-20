import os
from os import listdir
import CreateFeatureFile
import Classifier
from shutil import copyfile

serverDataPathBase = os.path.dirname(os.path.realpath(__file__)) + "/server-data/"

#### STEP 3: AUTH
files = [f for f in listdir(serverDataPathBase + "detect")]
files = filter(lambda x: x.endswith('.raw.csv'), files)

results = []

for file in files:
    detectUser = file.split("-")[0]
    detectFileName = file[:-8]

    if detectUser == "simon":
        continue

    auth_features_path = serverDataPathBase + "detect/"+detectFileName+".features.csv"
    CreateFeatureFile.main(serverDataPathBase + "detect/"+detectFileName+".raw.csv", \
                           auth_features_path,  4)

    model_path = os.path.dirname(os.path.realpath(__file__)) + "/server-data/models/"+detectUser+".model"
    authTrustScore = Classifier.predict(model_path, auth_features_path)
    print(detectUser + " > " + detectFileName + " > " + str(authTrustScore * 100) + "%")

    results.append([detectUser, detectFileName, authTrustScore])

print("Finished results:")
print(results)
