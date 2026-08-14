"""
Provides common utility functions for medication ATC codes, from INSPIRE dataset.
"""

from stopwatch import Stopwatch

import sys
import random
import os
import json
import csv


import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt



class ATCCode:
    def __init__(self, code, level1_name, level2_name, level3_name, level4_name, level5_name):
        self._code = code
        self._level1_name = level1_name
        self._level2_name = level2_name
        self._level3_name = level3_name
        self._level4_name = level4_name
        self._level5_name = level5_name

    def get_code(self):
        return self._code

    def get_level1_name(self):
        return self._level1_name
    def get_level2_name(self):
        return self._level2_name
    def get_level3_name(self):
        return self._level3_name
    def get_level4_name(self):
        return self._level4_name
    def get_level5_name(self):
        return self._level5_name

    def __str__(self):
        return self._code



def make_atc_code( atc_code,  atc_code_lookup):
    # 0123456
    # M01AE01
    level1_code = atc_code[0:1]
    level2_code = atc_code[0:3]
    level3_code = atc_code[0:4]
    level4_code = atc_code[0:5]
    level5_code = atc_code[0:7]

    level1_name = atc_code_lookup[level1_code]
    level2_name = atc_code_lookup[level2_code]
    level3_name = atc_code_lookup[level3_code]

    level4_name = None
    if level4_code in atc_code_lookup:
        level4_name = atc_code_lookup[level4_code]


    level5_name = None
    if level5_code in atc_code_lookup:
        level5_name = atc_code_lookup[level5_code]

    return ATCCode(atc_code, level1_name, level2_name, level3_name, level4_name, level5_name)


def parse_atc_code( atc_code,  atc_code_lookup):
    # 0123456
    # M01AE01
    level1_code = atc_code[0:1]
    level2_code = atc_code[0:3]
    level3_code = atc_code[0:4]
    level4_code = atc_code[0:5]
    level5_code = atc_code[0:7]

    level1_name = atc_code_lookup[level1_code]
    level2_name = atc_code_lookup[level2_code]
    level3_name = atc_code_lookup[level3_code]
    level4_name = atc_code_lookup[level4_code]
    level5_name = atc_code_lookup[level5_code]

    return f"{level1_name}|{level2_name}|{level3_name}|{level4_name}|{level5_name}"


def read_atc_codes(filepath='../codes/WHO_ATC-DDD_2024-07-31.csv'):
    atc_code_lookup = dict()
    with open(filepath, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        # row keys = [atc_code , atc_name, ddd, uom adm_r, note]
        for row in reader:
            atc_code = row["atc_code"]
            atc_name = row["atc_name"]
            atc_code_lookup[atc_code] = atc_name
    return atc_code_lookup

"""
Reads all packets from pcap into memory.
"""
def main():

    # if len(sys.argv) != 2:
    #     print("Usage <csv_file>")
    #     return
    # csv_file = sys.argv[1]

    atc_code_lookup = dict()
    with open("./dataset/WHO_ATC-DDD_2024-07-31.csv", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        # row keys = [atc_code , atc_name, ddd, uom adm_r, note]
        for row in reader:
            atc_code = row["atc_code"]
            atc_name = row["atc_name"]
            atc_code_lookup[atc_code] = atc_name



    print(parse_atc_code("M01AE01", atc_code_lookup))
    print(parse_atc_code("G03AC", atc_code_lookup))

    with open("./dataset/medications.csv", newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            drug_name1= row["drug_name"]
            route     = row["route"]
            drug_name2= row["drug_name2"]
            drug_name3= row["drug_name3"]
            atc_code1 = row["atc_code"]
            atc_code2 = row["atc_code2"]
            atc_code3 = row["atc_code3"]

            #if atc_code1.startswith("0"):
            #    print(f"INVALID CODE {atc_code1}")

            if len(atc_code1) > 4 and not atc_code1.startswith("0"):
                code = make_atc_code(atc_code1, atc_code_lookup)


    #medications_df = pd.read_csv("./dataset/WHO ATC-DDD 2024-07-31.csv")
    #print(medications_df)


if __name__ == "__main__":
    main()


"""
./dataset/medications.csv
         subject_id  chart_time       drug_name route drug_name2 drug_name3 atc_code atc_code2 atc_code3
0         117512122     2832985      pregabalin    po        NaN        NaN  N03AX16       NaN       NaN
1         117512122     2833610      pregabalin    po        NaN        NaN  N03AX16       NaN       NaN
2         117512122     2832985   levetiracetam    po        NaN        NaN  N03AX14       NaN       NaN
3         117512122     2833610   levetiracetam    po        NaN        NaN  N03AX14       NaN       NaN
4         117512122     2832985      famotidine    po        NaN        NaN  A02BA03       NaN       NaN
...             ...         ...             ...   ...        ...        ...      ...       ...       ...
9885567   127026720        2070       ofloxacin    ex        NaN        NaN  J01MA01       NaN       NaN
9885568   127026720        2550       ibuprofen    po   arginine        NaN  M01AE01   B05XB01       NaN
9885569   127026720        1340  levocetirizine    po        NaN        NaN  R06AE09       NaN       NaN
9885570   154044684         675      gentamicin    iv        NaN        NaN  J01GB03       NaN       NaN
9885571   154044684         675       cefazolin    iv        NaN        NaN  J01DB04       NaN       NaN

[9885572 rows x 9 columns]


The Anatomical Therapeutic Chemical (ATC) code is a unique identifier assigned to
a medicine based on the organ or system it affects and its therapeutic,
pharmacological, and chemical properties.

This classification system is maintained by the World Health Organization (WHO)
and is designed to aid in monitoring drug use and conducting research to
improve medication use.

Medications are categorized into different groups according to the organ
or system on which they act, their therapeutic intent, and their
chemical characteristics.

https://atcddd.fhi.no/atc_ddd_index_and_guidelines/atc_ddd_index/
https://atcddd.fhi.no/atc_ddd_index/

Useful, already crawled website for each level, easy lookup
https://github.com/fabkury/atcd

Example code: G03AC
G GENITO URINARY SYSTEM AND SEX HORMONES
G03 SEX HORMONES AND MODULATORS OF THE GENITAL SYSTEM
G03A HORMONAL CONTRACEPTIVES FOR SYSTEMIC USE
G03AC Progestogens


ATC code
All ATC levels are searchable.
A search will result in showing the exact substance/level and all ATC levels above (up to 1st ATC level).


Level 1: Anatomical Main Group (one letter)
Level 2: Therapeutic Main Group (two digits)
Level 3: Pharmacological Subgroup (one letter)
Level 4: Chemical/Therapeutic/Pharmacological Subgroup (one letter)
Level 5: Chemical Substance (two digits)

LEVEL 1
A ALIMENTARY TRACT AND METABOLISM
B BLOOD AND BLOOD FORMING ORGANS
C CARDIOVASCULAR SYSTEM
D DERMATOLOGICALS
G GENITO URINARY SYSTEM AND SEX HORMONES
H SYSTEMIC HORMONAL PREPARATIONS, EXCL. SEX HORMONES AND INSULINS
J ANTIINFECTIVES FOR SYSTEMIC USE
L ANTINEOPLASTIC AND IMMUNOMODULATING AGENTS
M MUSCULO-SKELETAL SYSTEM
N NERVOUS SYSTEM
P ANTIPARASITIC PRODUCTS, INSECTICIDES AND REPELLENTS
R RESPIRATORY SYSTEM
S SENSORY ORGANS
V VARIOUS


List of abbreviations

Unit (U) 

g	=	gram
mg	 	milligram
mcg	=	microgram
U	=	unit
TU	=	thousand units
MU	=	million units
mmol=	millimole
ml	=	milliliter (e.g. eyedrops)
    	  	           	    

Route of administration (Adm.R)

Implant	=	Implant
Inhal	= 	Inhalation 
Instill	= 	Instillation
N	=	nasal
O	=	oral
P	=	parenteral
R	=	rectal
SL	= 	sublingual/buccal/oromucosal
TD	=	transdermal
V	=	vaginal
"""


"""
./dataset/WHO ATC-DDD 2024-07-31.csv
     atc_code                                atc_name  ddd  uom adm_r             note
0           A         ALIMENTARY TRACT AND METABOLISM  NaN  NaN   NaN              NaN
1         A01             STOMATOLOGICAL PREPARATIONS  NaN  NaN   NaN              NaN
2        A01A             STOMATOLOGICAL PREPARATIONS  NaN  NaN   NaN              NaN
3       A01AA              Caries prophylactic agents  NaN  NaN   NaN              NaN
4     A01AA01                         sodium fluoride  1.1   mg     O  0.5 mg fluoride
...       ...                                     ...  ...  ...   ...              ...
7340  V10XX02              ibritumomab tiuxetan (90Y)  NaN  NaN   NaN              NaN
7341  V10XX03               radium (223Ra) dichloride  NaN  NaN   NaN              NaN
7342  V10XX04          lutetium (177Lu) oxodotreotide  NaN  NaN   NaN              NaN
7343  V10XX05  lutetium (177Lu) vipivotide tetraxetan  NaN  NaN   NaN              NaN
7344      V20                      SURGICAL DRESSINGS  NaN  NaN   NaN              NaN

[7345 rows x 6 columns]
"""
