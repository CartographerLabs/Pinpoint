folder = r"C:\Projects\Clean Pareler Data\parler_2020-01-06_posts-partial"
import csv
import os
import re
from html.parser import HTMLParser
from io import StringIO


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


iterator = 0
failed = 0
skipped = 0

list_of_data = []
files = os.listdir(folder)
for file_name in files:

    print("messages {} out of {} read | errors {}, successes {}, skipped {}".format(iterator, len(files), failed,
                                                                                    iterator - failed - skipped,
                                                                                    skipped))

    file_to_read = open(os.path.join(folder, file_name), "r", encoding='utf-8')
    data = file_to_read.read()

    if "author" in data:
        try:
            name = re.search(r'author--name">(.+)<\/span>', data).groups(1)[0]
            username = re.search(r'"author--username">(.+)<\/span>', data).groups(1)[0]
            timestamp = re.search(r'post--timestamp">(.+)<\/span>', data).groups(1)[0]
            message = re.search(r'<p>(.+)<\/p>', data).groups(1)[0]
            message = strip_tags(message)
            list_of_data.append({"name": name, "username": username, "timestamp": timestamp, "message": message})
        except:
            failed = failed + 1
    else:
        skipped = skipped + 1

    iterator = iterator + 1

with open('messages.csv', 'w', encoding='utf8', newline='') as output_file:
    fc = csv.DictWriter(output_file,
                        fieldnames=list_of_data[0].keys(),
                        )
    fc.writeheader()
    fc.writerows(list_of_data)

print("Found errors in {} out of {}".format(failed + skipped, iterator))
