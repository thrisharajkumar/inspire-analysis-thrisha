# inspire-analysis-thrisha
Analysis (e.g. peri-operative mortality prediction) of the INSPIRE dataset

## Description of CSV files

Taken from and copied here for ease of reference
https://www.nature.com/articles/s41597-024-03517-4


## Diagnosis
The ‘diagnosis’ table includes all diagnoses claimed by a physician in the 
form of ICD-10-CM from 6 months before the time zero to the discharge after 
the last operation, except for a set of pre-defined, sensitive diagnoses 
that needed to be removed (Table 2). Only the first three digits of the 
ICD-10-CM code and the relative time of diagnosis were presented. 
The most prevalent diagnosis was H26, which represents diseases associated 
with cataracts and presents in about 9,000 patients.

subject_id,chart_time,icd10_cm

## Vitals
The ‘vitals’ table includes all intraoperative vital signs, urine output,
fluid administration, estimated blood loss, anaesthesia machine 
settings such as inspiratory flow of O2 or concentration of anaesthesia gas, 
or ventilatory parameters, like tidal volume or peak inspiratory pressure
during operation. Variables measured by specialised devices, such as 
bispectral index and regional cerebral oxygen saturation, were also included. 
All variables were matched with subject_id and op_id, presented with 
value without the unit, and chart_time of 5-minute interval. 
Labels for the parameters are in the parameters table.

While most vital signs such as heart rate, respiratory rate, or peripheral 
oxygen saturation existed in most operations, variables measured by 
specialised devices were only in limited operation cases. 
Level of bispectral index and regional cerebral oxygen saturation 
were available in 65,236 cases (49.8%) and 205 cases (0.16%), respectively.


## Ward_vitals
While the ‘vitals’ table included intraoperative vital signs, the 
‘ward_vitals’ table included vital signs measured outside the operating room. 
From 6 months before the time 0 to the time of discharge after the 
last operation, all measured vital signs were included. The chart_time 
was expressed in 5-minute intervals, with the imputation with the median 
values for variables measured shorter than 5 minutes. Labels for 
the parameters are in the parameters table.

Regarding additional life-supporting devices, perioperative applications 
of ECMO, IABP, and CRRT were found in 166 (0.17%), 180 (0.18%), 
and 855 (0.86%) patients, respectively.

## Labs
Pre-defined laboratory variables were included in the ‘labs’ table with 
their value and chart_time. Laboratory results measured from 6 months 
before the time zero to 6 months after the last discharge were included. 
Labels for the parameters are in the parameters table.

Since our routine preoperative evaluation includes laboratory 
measurements for cell blood counts, renal and liver function tests, 
and coagulation tests within 6 months before the surgery, relevant 
laboratory variables were found in most cases. Intraoperative 
laboratory measurements were primarily restricted to point-of-care 
testing for arterial blood gas analysis among patients with 
arterial catheters, with a maximum interval of 2 hours between measurements.


## Medications
The ‘medications’ table includes data on medications administrated 
between 6 months before the time 0 and the time of the last discharge. 
Information captured in the table includes subject_id, chart_time 
as the time of the drug administered, drug_name as the ingredient 
name, and route as the route of drug administered were included 
in the ‘medications’ table. Fluid administrations such as 
balanced crystalloid, normal saline, or dextrose solution in 
general wards were not included. To avoid the risk of re-identification 
by using rarely administered medications, chemotherapy, immunotherapy, 
research drugs, and medications administered to less than 100 patients 
were excluded.

As a result, 9,926,795 administrations of medication were recorded 
from 99,807 patients. Among these, 1,376 unique combinations of 
drugs and administration routes were identified, 
comprising 1,238 distinct types of drugs.


## Parameters
The ‘parameters’ table includes physical units and the human-readable 
description of the parameters in the labs, vitals, and ward_vitals tables.



