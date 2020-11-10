# Python program to convert
# JSON file to CSV


import json
import csv


# Opening JSON file and loading the data
# into the variable data
with open('335850455-1604057017.json') as json_file:
    data = json.load(json_file)

texts = data['texts']

# now we will open a file for writing
data_file = open('data_file.csv', 'w')

# create the csv writer object
csv_writer = csv.writer(data_file)

# Counter variable used for writing
# headers to the CSV file
count = 0

for text in texts:
    if count == 0:

        # Writing headers of CSV file
        header = text.keys()
        csv_writer.writerow(header)
        count += 1

    # Writing data of CSV file
    csv_writer.writerow(text.values())

data_file.close()
