import os
from os import listdir
from os.path import isfile, join
import CreateFeatureFile
import Classifier
from shutil import copyfile

serverDataPathBase = os.path.dirname(os.path.realpath(__file__)) + "/server-data/"
serverDataRecordsPath = serverDataPathBase + "records"

users = [f for f in listdir(serverDataRecordsPath)]
skip_users = []

def fix_feature(c, s):
    chunks = s.split(",")
    chunks[0] = str(c)
    return ",".join(chunks)

#### STEP 1: extract
for user in users:
    userBasePath = join(serverDataRecordsPath, user)
    allRecords = listdir(userBasePath)
    records = filter(lambda x: x.endswith('.raw.csv'), allRecords)
    featuresHead = ""
    allFeatures = []
    for record in records:
        recordName = record[:-8]
        recordBasePath = join(userBasePath, recordName)
        recordPath = join(userBasePath, recordName + ".raw.csv")
        featuresPath = join(userBasePath, recordName + ".features.csv")
        CreateFeatureFile.main(recordPath, featuresPath, 4)
        # # Old way
        # copyfile(featuresPath, serverDataPathBase + "features/" + user + ".csv")
        with open(featuresPath) as f:
            lines = f.readlines()
            if len(allFeatures) == 0:
                featuresHead = lines[0].replace('\n', '')
                allFeatures = lines[1:]
            else:
                allFeatures = allFeatures + lines[1:]

    beforeCleanLen = len(allFeatures)
    # allFeatures = list(filter(lambda x: "inf" not in x, allFeatures))
    if beforeCleanLen != len(allFeatures):
        print("WARNING inf found for " + user)
    if len(allFeatures) <= 1:
        print("WARNING no features for " + user)
        skip_users.append(user)
    f = open(serverDataPathBase + "features/" + user + ".csv", "w")


    allFeatures = map(lambda x: fix_feature(x[0], x[1]), enumerate(allFeatures))

    f.write(featuresHead + "\n" + "\n".join(map(lambda x: x.replace('\n', ''), allFeatures)))
    f.close()


stats = []
#### STEP 2: CREATE MODELS
for user in users:
    if user in skip_users:
        print("!!! SKIPPED MODEL FOR: " + user)
        continue

    print("CREATING MODEL FOR: " + user)
    generalScore = Classifier.main(
        os.path.dirname(os.path.realpath(__file__)) + "/server-data/features/", \
        os.path.dirname(os.path.realpath(__file__)) + "/server-data/models/", \
        user
    )

    #only to see stats what is best not used on server
    scores = Classifier.getScoreForAll(
        os.path.dirname(os.path.realpath(__file__)) + "/server-data/features/",
        user
    )
    stats.append([user, scores])

print("Done\n")
print("Skipped users: ")
print(skip_users)
print("stats: ")
print(stats)

print("")
print("===================================================")
for s in stats:
    line = "X"
    for x in s[1]:
        far = x[2]
        frr = x[3]
        line += " & " + str('%.2f' % far)
    print(line + "\\\\ \\hline")

print("===================================================")

for s in stats:
    line = "X"
    for x in s[1]:
        far = x[2]
        frr = x[3]
        line += " & " + str('%.2f' % frr)
    print(line + "\\\\ \\hline")