import numpy as np
import pandas as pd


class data:
    def __init__(self):
        dataList = [
            pd.read_csv("TrainingData\english.csv"),
            pd.read_csv("TrainingData\spanish.csv"),
            pd.read_csv("TrainingData\swahili.csv"),
            pd.read_csv("TrainingData\polish.csv"),
            pd.read_csv("TrainingData\japanese.csv")
        ]

        dataLabels = [
            pd.DataFrame({"Label": ["eng" for i in range(len(dataList[0]))]}), 
            pd.DataFrame({"Label": ["spn" for i in range(len(dataList[1]))]}), 
            pd.DataFrame({"Label": ["swh" for i in range(len(dataList[2]))]}),
            pd.DataFrame({"Label": ["pol" for i in range(len(dataList[3]))]}),
            pd.DataFrame({"Label": ["jap" for i in range(len(dataList[4]))]}),
        ]

        self.trainingData = [pd.concat(
            [d, l], ignore_index=True, axis=1) for d,l in zip(dataList,dataLabels)]
        for d in self.trainingData:
            d.columns = ["Word", "Label"]
        #labelList = pd.DataFrame(
        #    {"Label": })

        #self.trainingData.columns = ["Word", "Label"]
        #self.trainingData = self.trainingData.sample(
        #    frac=1).reset_index(drop=True)

    def get(self, num):
        samp = pd.concat([d.sample(n=num) for d in self.trainingData])
        samp = samp.sample(frac=1)
        word = [wordToVec(w) for w in samp['Word']]
        label = [labToVec(l) for l in samp['Label']]
        return (word, label, samp['Word'].iloc[0])


def wordToVec(w):
    vec = np.zeros((26 * 18))
    for i, ch in enumerate(w):
        vec[(ord(ch) - 97 + i * 26)] = 1
    return vec


def labToVec(l):
    lab = {
        "spn": 0,
        "eng": 0,
        "swh": 0,
        "pol":0,
        "jap":0,
    }
    lab[l] = 1
    return np.array(np.array(list(lab.values())))
