import json

messages_from_file = []

with open(r'C:\Projects\Pinpoint\outputs\training_features.json') as json_file:
    data = json.load(json_file)
    for message_entity in data:
        messages_from_file.append(message_entity)

comparison_dict = {}
count = 0
for message_entity in messages_from_file:
    count = count + 1
    for username in message_entity.keys():
        message = message_entity[username]
        print("Processing message {} of {}".format(count, len(messages_from_file)))

        centrality = message["centrality"]
        is_extreamist = str(message["is_extremist"]).lower()

        if centrality != 0:
            if is_extreamist in comparison_dict.keys():
                comparison_dict[is_extreamist] = comparison_dict[is_extreamist] + 1
            else:
                comparison_dict[is_extreamist] = 1


total_is_extreamist = comparison_dict["true"]
total_not_extreamist = comparison_dict["false"]
total = total_is_extreamist + total_not_extreamist

is_extreamist_percentage = (total_is_extreamist/ total) * 100
not_extreamist_percentage = (total_not_extreamist/total) * 100

print("Of the posts with a centrality higher than 0, {}% were extreamist posts and {}% were non-extreamist posts")

output_file = open("centrality-count.txt", "w", encoding="utf8")
json.dump(comparison_dict, output_file)