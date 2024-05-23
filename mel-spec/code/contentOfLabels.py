import pickle
import pprint

obj = pickle.load(open("data.pkl", "rb"))

with open("data.txt", "a") as f:
    pprint.pprint(obj, stream=f)