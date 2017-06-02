import csv
import codecs
import pprint
import re
import xml.etree.cElementTree as ET
from collections import defaultdict

brooklyn = "brooklyn_new-york.osm"
sample = "sample.osm"

expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Highway", "Parkway", "Road", "Extension",
           "Path", "Park", "Plaza", "Walk", "Square", "Piers", "Lane", "Center"]

street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

def get_element(osm_file):
    '''Yield element if it is the right type of tag. This is from Myles. 
    Source:https://discussions.udacity.com/t/problem-with-finding-tags-no-memory/195154/2
    '''
    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end':
            yield elem
            root.clear()

def count_tags(osm_file):
    """This function returns what tags and number 
    of tags are in the osm file."""
    tags = {}
    for elem in get_element(osm_file):
        if elem.tag in tags.keys():
            tags[elem.tag] += 1
        else:
            tags[elem.tag] = 1
        elem.clear()
    return tags


def is_street_name(elem):
    """Returns elements that contain street names."""
    return (elem.attrib['k'] == "addr:street")

def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)

def audit_street(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for i, elem in enumerate(get_element(osmfile)):
        if elem.tag in ["node" ,"way"]:
            for tag in elem.iter('tag'):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
    osm_file.close()
    return street_types

def is_zipcode(elem):
    return (elem.attrib['k'] == "addr:postcode")

def audit_zipcode(invalid_zipcodes, zipcode):
    '''Returns zipcodes that are not in the Brooklyn area.'''
    threeDigits = zipcode[0:3]
    if threeDigits != 112 or not threeDigits.isdigit():
        invalid_zipcodes[threeDigits].add(zipcode)
           
def audit_zip(osmfile):
    '''Returns a dictionary of zipcodes in the osm file'''
    osm_file = open(osmfile, "r")
    invalid_zipcodes = collections.defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_zipcode(tag):
                    audit_zipcode(invalid_zipcodes,tag.attrib['v'])

    return invalid_zipcodes

def is_phone(elem):
    '''Search elements that have phone numbers.'''
    return(elem.tag == 'tag') and (elem.attrib['k'] == "phone" or elem.attrib['k']=='fax')

def audit_phone(osmfile):
    '''Takes phone numbers from osm file and output a set of phone numbers'''
    osm_file = open(osmfile, "r")
    phone_nums = set()
    for event, elem in ET.iterparse(osmfile):
        if elem.tag in ["node" ,"way"]:
             for tag in elem.iter("tag"):
                if tag.attrib['k'] == "phone":
                    phone_nums.add(tag.attrib['v'])
    return phone_nums

#if __name__ == "__main__":
#   tags = count_tags(sample)
#   pprint.pprint(tags)
#---------------------------------------
#   bk_street_types = audit_street(sample)
#   pprint.pprint(dict(bk_street_types))
#---------------------------------------
#   bk_zipcode = audit_zip(brooklyn)
#   pprint.pprint(dict(bk_zipcode))
#---------------------------------------
#   bk_phone = audit_phone(brooklyn)
#   pprint.pprint(bk_phone) 
