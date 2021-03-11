import xml.etree.ElementTree as ET
import re


def transform_content(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    children = root.getchildren()
    if len(children) == 0:
        print('Error: Wrong format for file {}'.format(filename))
        return None
    # replace \n - to put together the sentences
    content = re.sub(r'\n', ' ', children[0].text)
    content = " ".join(content.split())
    return content


content = transform_content("RAW/1.txt")

f = open("INPUT/1.txt", "w")
f.write(content)
