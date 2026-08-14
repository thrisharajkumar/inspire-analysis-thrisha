"""
ICD-10-CM (International Classification of Diseases, 10th Edition, Clinical Modification)
is a code set used to classify diagnoses in both the inpatient and outpatient setting,
 while ICD-10-PCS (International Classification of Diseases, 10th Edition,
 Procedure Coding System) is a coding system used to classify procedures performed
 in the inpatient setting.

There is no single list of ICD-10 codes. Instead, ICD-10-CM and ICD-10-PCS are
separate code sets, each with its own specific purpose. ICD-10-CM is used for
diagnosing patients, while ICD-10-PCS is used for classifying procedures.

The ICD-10-CM code set is maintained by the National Center for Health Statistics (NCHS),
while the ICD-10-PCS code set is maintained by the Centers for Medicare and Medicaid Services (CMS).

The ICD-10-CM code set is used in both the inpatient and outpatient setting,
while the ICD-10-PCS code set is used only in the inpatient setting.

The ICD-10-CM code set is based on ICD-10, the system used to code and
classify mortality data from death certificates, while the ICD-10-PCS code set
is a new procedure coding system developed by CMS for inpatient procedures.

The ICD-10-CM code set is used to code and classify medical diagnoses,
while the ICD-10-PCS code set is used to code and classify procedures
performed in the inpatient setting.
"""

"""
See https://www.cms.gov/medicare/coding-billing/icd-10-codes
"""

"""
The following code is taken from https://github.com/bryand1/icd10-cm/blob/master/icd10/__init__.py
PLEASE CITE THEM.
The copy is done to avoid dependency issues (yet another library to install and fail).
"YALIF" (Yet Another Library Incompatible with Future Versions) 
Based on a commit, believe this is 2023
"Update to icd10 2023"
NB: the codes are modified every year, 2026 is available as of 21 July 2025.
    but not sure how to get into the JSON format used by the github code.

The JSON format is fairly straight forward with three fields (code, billable, description),
with chapter, block, and block_description derived.
{
  "A00": [
    false,
    "Cholera"
  ],
  "A000": [
    true,
    "Cholera due to Vibrio cholerae 01, biovar cholerae"
  ],
...
"""

"""
ICD-10 CM

ICD-10 is the 10th revision of the International Statistical Classification of Diseases and
Related Health Problems (ICD), a medical classification list by the World Health Organization (WHO).
It contains codes for diseases, signs and symptoms, abnormal findings, complaints, social circumstances,
and external causes of injury or diseases.
"""
import gzip
import json
import os
from typing import Optional


# https://en.wikipedia.org/wiki/ICD-10#List
chapters = [
    ('I', 'A00-B99', 'Certain infectious and parasitic diseases'),
    ('II', 'C00-D48', 'Neoplasms'),
    ('III', 'D50-D89', 'Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism'),
    ('IV', 'E00-E90', 'Endocrine, nutritional and metabolic diseases'),
    ('V', 'F00-F99', 'Mental and behavioural disorders'),
    ('VI', 'G00-G99', 'Diseases of the nervous system'),
    ('VII', 'H00-H59', 'Diseases of the eye and adnexia'),
    ('VIII', 'H60-H95', 'Diseases of the ear and mastoid process'),
    ('IX', 'I00-I99', 'Diseases of the circulatory system'),
    ('X', 'J00-J99', 'Diseases of the respiratory system'),
    ('XI', 'K00-K93', 'Diseases of the digestive system'),
    ('XII', 'L00-L99', 'Diseases of the skin and subcutaneous tissue'),
    ('XIII', 'M00-M99', 'Diseases of the musculoskeletal system and connective tissue'),
    ('XIV', 'N00-N99', 'Diseases of the genitourinary system'),
    ('XV', 'O00-O99', 'Pregnancy, childbirth and the puerperium'),
    ('XVI', 'P00-P96', 'Certain conditions originating in the perinatal period'),
    ('XVII', 'Q00-Q99', 'Congenital malformations, deformations and chromosomal abnormalities'),
    ('XVIII', 'R00-R99', 'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified'),
    ('XIX', 'S00-T98', 'Injury, poisoning and certain other consequences of external causes'),
    ('XX', 'V01-Y98', 'External causes of morbidity and mortality'),
    ('XXI', 'Z00-Z99', 'Factors influencing health status and contact with health services'),
    ('XXII', 'U00-U99', 'Codes for special purposes'),
]


class ICD10:

    def __init__(self, code: str, billable: bool, description: str):
        self.code = code
        self.billable = billable
        self.categorical = not billable
        self.description = description
        self._chapter = None
        self._block = None
        self._block_description = None

    @property
    def chapter(self) -> str:
        if self._chapter is not None:
            return self._chapter
        self._find_chapter()
        return self._chapter

    @property
    def block(self) -> str:
        if self._block is not None:
            return self._block
        self._find_chapter()
        return self._block

    @property
    def block_description(self) -> str:
        if self._block_description is not None:
            return self._block_description
        self._find_chapter()
        return self._block_description

    def __str__(self):
        if len(self.code) > 3:
            return self.code[:3] + '.' + self.code[3:]
        else:
            return self.code

    def __repr__(self):
        return "<ICD10: %s>" % self

    def __hash__(self):
        return hash(self.code)

    def in_chapter(self, block: str, icd10: str) -> bool:
        alpha, numeric = ord(icd10[0]), int(icd10[1:3].lstrip('0'))
        sblock, eblock = block.split('-')  # A00-B99
        salpha, snumeric = ord(sblock[0]), int(sblock[1:].lstrip('0') or 0)
        ealpha, enumeric = ord(eblock[0]), int(eblock[1:].lstrip('0') or 0)
        return salpha <= alpha <= ealpha and snumeric <= numeric <= enumeric

    def _find_chapter(self):
        for chapter, block, description in chapters:
            if self.in_chapter(block, self.code):
                self._chapter = chapter
                self._block = block
                self._block_description = description
                break





class ICD10Repository:
    def __init__(self, json_codes):
        self._json_codes = json_codes

    def exists(self, s: str) -> bool:
        """
        exists("T50.B")
        True
        exists("A99.8")
        False
        """
        if not s:
            return False
        return bool(self._json_codes.get(s.replace('.', ''), False))


    def find(self, s: str) -> Optional[ICD10]:
        """
        find("A02.1")
        <ICD10: A02.1>
        """
        if not s:
            return
        k = s.replace('.', '')
        v = self._json_codes.get(k)
        if v is None:
            return
        billable, description = v
        return ICD10(k, billable, description)

def make_repository(gz_filepath='../codes/icd10.json.gz'):
    # here = os.path.dirname(os.path.abspath(__file__))
    # fh = gzip.open(os.path.join(gz_filepath, 'icd10.json.gz'))
    fh = gzip.open(gz_filepath)
    json_codes = json.load(fh)
    fh.close()

    return ICD10Repository(json_codes)



def main():
    # here = os.path.dirname(os.path.abspath(__file__)) # JHP to use relative path to find JSON file
    icd10_repo = make_repository()

    code = icd10_repo.find("J20.0")
    print(code.description)  # Acute bronchitis due to Mycoplasma pneumoniae
    if code.billable:
        print(code, "is billable")  # J20.0 is billable

    print(code.chapter)  # X
    print(code.block)  # J00-J99
    print(code.block_description)  # Diseases of the respiratory system


if __name__ == "__main__":
    main()