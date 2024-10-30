import json
case_file_path = "../Case_topic.json"
description_file_path = "../Descriptions.json"

with open(case_file_path) as f:
    case_json = json.load(f)

with open(description_file_path) as f:
    description_json = json.load(f)

# this has image information
print(len(description_json))
# this has patient information
print(len(case_json))

#inspect keys of the json
#dict_keys(['Type', 'U_id', 'image', 'Description', 'Location', 'Location Category'])
print(description_json[0].keys())

#dict_keys(['U_id', 'TAC', 'MRI', 'Case', 'Topic'])
# TAC holds the image_ids
print(case_json[0].keys())

# Idea:
# remove any dimensions text / location text <- by prompting
# for description we will track:
# • Caption <- what we see in image 
# • Gender
# • Modality [Plane]
# • Location [Location Category]
# for case we will track:
# • Findings <- what we saw
# • Differential Diagnosis <- what we considered
# • Case Diagnosis <- result diagnosis ?
# • Diagnosis by <- Method of diagnosis
# • Treatment and follow up <- what they might be told by the doctor
# what [Title] usually mean ans disease discussion

# we will actually do the join operation later

description_dict = dict()
for i in description_json:
    img = i["image"]
    description_dict[img] = {
        "caption":i["Description"]["Caption"],
        "modality":i["Description"]["Modality"],
        "plane":i["Description"]["Plane"],
        "gender":i["Description"]["Sex"],
        "location":i["Location"],
        "location_category":i["Location Category"]
            } 

json.dump(description_dict, open("description_dict.json", "w"))
