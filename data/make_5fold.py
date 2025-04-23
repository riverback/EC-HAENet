import os
import json
import numpy as np

def help_sum(data):
    count = 0
    for d in data:
        count += len(d[1]["img_id"])
    return count

original_data_json = "Esophageal-Cancer-Dataset/patient_data.json"
with open(original_data_json, "r", encoding="utf-8") as f:
    data = json.load(f)


Cancer_Patient = []
MPR_Patient = []
PCR_Patient = []

for patient_id, patient_data in data.items():
    if patient_data["patient_label"] == "Cancer":
        Cancer_Patient.append((patient_id, patient_data))
    elif patient_data["patient_label"] == "MPR":
        MPR_Patient.append((patient_id, patient_data))
    elif patient_data["patient_label"] == "PCR":
        PCR_Patient.append((patient_id, patient_data))
    else:
        raise ValueError("Unknown label: {}".format(patient_data["label"]))
print("Cancer_Patient: {}".format(len(Cancer_Patient)))
print("MPR_Patient: {}".format(len(MPR_Patient)))
print("PCR_Patient: {}".format(len(PCR_Patient)))
print(type(Cancer_Patient[0]))

# Cancer 5 fold
Cancer_Patient = sorted(Cancer_Patient, key=lambda x: len(x[1]["img_id"]))[::-1]
print("Cancer", len(Cancer_Patient[0][1]["img_id"]), len(Cancer_Patient[-1][1]["img_id"]))
Cancer_fold_1 = []
Cancer_fold_2 = []
Cancer_fold_3 = []
Cancer_fold_4 = []
Cancer_fold_5 = []
for i in range(len(Cancer_Patient)):
    if i % 5 == 0:
        Cancer_fold_1.append(Cancer_Patient[i])
    elif i % 5 == 1:
        Cancer_fold_2.append(Cancer_Patient[i])
    elif i % 5 == 2:
        Cancer_fold_3.append(Cancer_Patient[i])
    elif i % 5 == 3:
        Cancer_fold_4.append(Cancer_Patient[i])
    elif i % 5 == 4:
        Cancer_fold_5.append(Cancer_Patient[i])
    else:
        raise ValueError("Unknown fold: {}".format(i))
# balance the number of images
print("Cancer_fold_1: {}, img: {}".format(len(Cancer_fold_1), help_sum(Cancer_fold_1)))
print("Cancer_fold_2: {}, img: {}".format(len(Cancer_fold_2), help_sum(Cancer_fold_2)))
print("Cancer_fold_3: {}, img: {}".format(len(Cancer_fold_3), help_sum(Cancer_fold_3)))
print("Cancer_fold_4: {}, img: {}".format(len(Cancer_fold_4), help_sum(Cancer_fold_4)))
print("Cancer_fold_5: {}, img: {}".format(len(Cancer_fold_5), help_sum(Cancer_fold_5)))
Cancer_fold_5.append(Cancer_fold_1.pop())
Cancer_fold_5.append(Cancer_fold_1.pop())
Cancer_fold_5.append(Cancer_fold_2.pop())
Cancer_fold_4.append(Cancer_fold_1.pop())
print("After balance the number of images")
print("Cancer_fold_1: {}, img: {}".format(len(Cancer_fold_1), help_sum(Cancer_fold_1)))
print("Cancer_fold_2: {}, img: {}".format(len(Cancer_fold_2), help_sum(Cancer_fold_2)))
print("Cancer_fold_3: {}, img: {}".format(len(Cancer_fold_3), help_sum(Cancer_fold_3)))
print("Cancer_fold_4: {}, img: {}".format(len(Cancer_fold_4), help_sum(Cancer_fold_4)))
print("Cancer_fold_5: {}, img: {}".format(len(Cancer_fold_5), help_sum(Cancer_fold_5)))


# 记录patient_id即可
fold_data = {
    "fold_1": [],
    "fold_2": [],
    "fold_3": [],
    "fold_4": [],
    "fold_5": []
}

fold_data["fold_1"] = [patient_id for patient_id, _ in Cancer_fold_1]
fold_data["fold_2"] = [patient_id for patient_id, _ in Cancer_fold_2]
fold_data["fold_3"] = [patient_id for patient_id, _ in Cancer_fold_3]
fold_data["fold_4"] = [patient_id for patient_id, _ in Cancer_fold_4]
fold_data["fold_5"] = [patient_id for patient_id, _ in Cancer_fold_5]

# MPR 5 fold
MPR_Patient = sorted(MPR_Patient, key=lambda x: len(x[1]["img_id"]))[::-1]
print("MPR", len(MPR_Patient[0][1]["img_id"]), len(MPR_Patient[-1][1]["img_id"]))
MPR_fold_1 = []
MPR_fold_2 = []
MPR_fold_3 = []
MPR_fold_4 = []
MPR_fold_5 = []
for i in range(len(MPR_Patient)):
    if i % 5 == 0:
        MPR_fold_1.append(MPR_Patient[i])
    elif i % 5 == 1:
        MPR_fold_2.append(MPR_Patient[i])
    elif i % 5 == 2:
        MPR_fold_3.append(MPR_Patient[i])
    elif i % 5 == 3:
        MPR_fold_4.append(MPR_Patient[i])
    elif i % 5 == 4:
        MPR_fold_5.append(MPR_Patient[i])
    else:
        raise ValueError("Unknown fold: {}".format(i))
# balance the number of images
print("MPR_fold_1: {}, img: {}".format(len(MPR_fold_1), help_sum(MPR_fold_1)))
print("MPR_fold_2: {}, img: {}".format(len(MPR_fold_2), help_sum(MPR_fold_2)))
print("MPR_fold_3: {}, img: {}".format(len(MPR_fold_3), help_sum(MPR_fold_3)))
print("MPR_fold_4: {}, img: {}".format(len(MPR_fold_4), help_sum(MPR_fold_4)))
print("MPR_fold_5: {}, img: {}".format(len(MPR_fold_5), help_sum(MPR_fold_5)))
MPR_fold_5.append(MPR_fold_1.pop())
MPR_fold_5.append(MPR_fold_2.pop())
print("After balance the number of images")
print("MPR_fold_1: {}, img: {}".format(len(MPR_fold_1), help_sum(MPR_fold_1)))
print("MPR_fold_2: {}, img: {}".format(len(MPR_fold_2), help_sum(MPR_fold_2)))
print("MPR_fold_3: {}, img: {}".format(len(MPR_fold_3), help_sum(MPR_fold_3)))
print("MPR_fold_4: {}, img: {}".format(len(MPR_fold_4), help_sum(MPR_fold_4)))
print("MPR_fold_5: {}, img: {}".format(len(MPR_fold_5), help_sum(MPR_fold_5)))

fold_data["fold_1"] += [patient_id for patient_id, _ in MPR_fold_1]
fold_data["fold_2"] += [patient_id for patient_id, _ in MPR_fold_2]
fold_data["fold_3"] += [patient_id for patient_id, _ in MPR_fold_3]
fold_data["fold_4"] += [patient_id for patient_id, _ in MPR_fold_4]
fold_data["fold_5"] += [patient_id for patient_id, _ in MPR_fold_5]

# PCR 5 fold
PCR_Patient = sorted(PCR_Patient, key=lambda x: len(x[1]["img_id"]))[::-1]
print("PCR", len(PCR_Patient[0][1]["img_id"]), len(PCR_Patient[-1][1]["img_id"]))
PCR_fold_1 = []
PCR_fold_2 = []
PCR_fold_3 = []
PCR_fold_4 = []
PCR_fold_5 = []
for i in range(len(PCR_Patient)):
    if i % 5 == 0:
        PCR_fold_1.append(PCR_Patient[i])
    elif i % 5 == 1:
        PCR_fold_2.append(PCR_Patient[i])
    elif i % 5 == 2:
        PCR_fold_3.append(PCR_Patient[i])
    elif i % 5 == 3:
        PCR_fold_4.append(PCR_Patient[i])
    elif i % 5 == 4:
        PCR_fold_5.append(PCR_Patient[i])
    else:
        raise ValueError("Unknown fold: {}".format(i))
# balance the number of images
print("PCR_fold_1: {}, img: {}".format(len(PCR_fold_1), help_sum(PCR_fold_1)))
print("PCR_fold_2: {}, img: {}".format(len(PCR_fold_2), help_sum(PCR_fold_2)))
print("PCR_fold_3: {}, img: {}".format(len(PCR_fold_3), help_sum(PCR_fold_3)))
print("PCR_fold_4: {}, img: {}".format(len(PCR_fold_4), help_sum(PCR_fold_4)))
print("PCR_fold_5: {}, img: {}".format(len(PCR_fold_5), help_sum(PCR_fold_5)))
PCR_fold_5.append(PCR_fold_1.pop())
PCR_fold_5.append(PCR_fold_1.pop())
PCR_fold_4.append(PCR_fold_2.pop())
PCR_fold_4.append(PCR_fold_2.pop())
print("After balance the number of images")
print("PCR_fold_1: {}, img: {}".format(len(PCR_fold_1), help_sum(PCR_fold_1)))
print("PCR_fold_2: {}, img: {}".format(len(PCR_fold_2), help_sum(PCR_fold_2)))
print("PCR_fold_3: {}, img: {}".format(len(PCR_fold_3), help_sum(PCR_fold_3)))
print("PCR_fold_4: {}, img: {}".format(len(PCR_fold_4), help_sum(PCR_fold_4)))
print("PCR_fold_5: {}, img: {}".format(len(PCR_fold_5), help_sum(PCR_fold_5)))

fold_data["fold_1"] += [patient_id for patient_id, _ in PCR_fold_5]
fold_data["fold_2"] += [patient_id for patient_id, _ in PCR_fold_4]
fold_data["fold_3"] += [patient_id for patient_id, _ in PCR_fold_3]
fold_data["fold_4"] += [patient_id for patient_id, _ in PCR_fold_2]
fold_data["fold_5"] += [patient_id for patient_id, _ in PCR_fold_1]

for k, v in fold_data.items():
    print(k, len(v))

with open("Esophageal-Cancer-Dataset/fold_data.json", "w", encoding="utf-8") as f:
    json.dump(fold_data, f, indent=4, ensure_ascii=False)

