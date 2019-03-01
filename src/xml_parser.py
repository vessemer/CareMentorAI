import pandas as pd
import xml.etree.ElementTree as et


def parse_occlusion(obj):
    occluded = obj.find('{http://www.w3.org/1999/xhtml}occluded')
    return occluded.text


def parse_polygon(obj):
    polygon = obj.find('{http://www.w3.org/1999/xhtml}polygon')
    points = polygon.findall('{http://www.w3.org/1999/xhtml}pt')

    coords = list()
    for point in points:
        x = point.find('{http://www.w3.org/1999/xhtml}x')
        y = point.find('{http://www.w3.org/1999/xhtml}y')
        x, y = int(x.text), int(y.text)
        coords.append((x, y))
    return coords


def compose_dataframe(data, row):
    data = pd.DataFrame(data)
    data['id'] = row.ID
    data['filename'] = row['Файлы']
    data['case'] = row['Кейс']
    return data


def parse_xml(row):
    xml_tree = et.fromstring(row.XML)
    objects = xml_tree.findall('{http://www.w3.org/1999/xhtml}object')

    data = list()

    for obj in objects:
        occlusion = parse_occlusion(obj)
        coords = parse_polygon(obj)

        data.append({
            'occluded': occlusion, 
            'coords': coords, 
        })
    return compose_dataframe(data, row)
