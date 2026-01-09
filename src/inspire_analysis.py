"""
Analyses csv files, from INSPIRE dataset.
"""
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt



def count_lines(filename):
    """
    Reads a file line by line using standard file operations.
    Each line is returned as a raw string.
    """
    count = 0
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()  # remove leading/trailing whitespace
            # Replace with your processing logic
            count += 1
            if count < 10 :
                print(line)
    return count




def analyse_vitals():
    # Load data
    vitals_df = pd.read_csv("C:\\Users\\pc\\Desktop\\dataset\\vitals.csv")
    operations_df = pd.read_csv("C:\\Users\\pc\\Desktop\\dataset\\operations.csv")

    # Convert chart_time to numeric or datetime if needed
    vitals_df['chart_time'] = pd.to_numeric(vitals_df['chart_time'], errors='coerce')

    vitals_df.sort_values(by="item_name", inplace=True)

    # --- Plot 1: Count of each vital sign type ---
    plt.figure(figsize=(20, 8))
    sns.countplot(data=vitals_df, x="item_name", order=vitals_df['item_name'].value_counts().index)
    plt.title("Distribution of Vital Sign Types")
    plt.xlabel("Vital Sign Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("fig_vitals_types.pdf", dpi=300)  # Save as publication-quality PDF
    plt.show()

    # --- Plot 2: Distribution of vital values by type ---
    plt.figure(figsize=(24, 10))
    sns.boxplot(data=vitals_df, x="item_name", y="value")
    plt.title("Distribution of Vital Sign Values by Type")
    plt.xlabel("Vital Sign Type")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.savefig("fig_vitals_values.pdf", dpi=300)  # Save as publication-quality PDF
    plt.tight_layout()
    plt.show()

    # # --- Plot 3: Trends in vitals over chart_time (optional) ---
    # plt.figure(figsize=(12, 6))
    # sns.lineplot(data=vitals_df[vitals_df['item_name'] == 'vt'], x='chart_time', y='value', hue='subject_id', legend=False)
    # plt.title("Tidal Volume (VT) Over Time for Each Subject")
    # plt.xlabel("Chart Time")
    # plt.ylabel("VT Value")
    # plt.tight_layout()
    # plt.show()
    # 
    # --- Join vitals with operations on op_id ---
    merged_df = pd.merge(vitals_df, operations_df, on="op_id", how="inner")
    # 
    # # --- Plot 4: Vital sign values vs ICU in time (if available) ---
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(data=merged_df, x="icuin_time", y="value", hue="item_name")
    # plt.title("Vital Values vs ICU In Time")
    # plt.xlabel("ICU In Time")
    # plt.ylabel("Vital Value")
    # plt.tight_layout()
    # plt.show()

    # --- Plot 5: Vital signs for patients who died in hospital ---
    deaths_df = merged_df[merged_df["inhosp_death_time"].notna()]
    deaths_df.sort_values(by="item_name", inplace=True)

    plt.figure(figsize=(20, 8))
    sns.boxplot(data=deaths_df, x="item_name", y="value")
    plt.title("Vitals for Patients with In-Hospital Deaths")
    plt.xlabel("Vital Sign Type")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("fig_vitals_deaths.pdf", dpi=300)  # Save as publication-quality PDF
    plt.show()

    survived_df = merged_df[merged_df["inhosp_death_time"].isna()]
    survived_df.sort_values(by="item_name", inplace=True)

    plt.figure(figsize=(20, 8))
    sns.boxplot(data=survived_df, x="item_name", y="value")
    plt.title("Vitals for Patients with In-Hospital Survived")
    plt.xlabel("Vital Sign Type")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("fig_vitals_survived.pdf", dpi=300)  # Save as publication-quality PDF
    plt.show()



"""
Reads all packets from pcap into memory.
"""
def main():

    # if len(sys.argv) != 2:
    #     print("Usage <csv_file>")
    #     return
    # csv_file = sys.argv[1]

    #medications_df = pd.read_csv("C:\\Users\\pc\\Desktop\\dataset\\medications.csv")
    #print(medications_df)

    analyse_vitals()

    # df = read_operations(department={"GS"})
    # print(df)

    # stopwatch = Stopwatch()
    # #df = pd.DataFrame(csv_file)
    # df = pd.read_csv(csv_file)
    # #df = df.sort_values(['model_name'])  # natural in a way
    # print(df)
    # print(f"File {csv_file}, (rows, cols)={df.shape}, took {stopwatch.elapsedTime() / 60:.2f} minutes")
    #
    # #lines_read = count_lines(csv_file)
    # #print(f"File {csv_file}, read {lines_read} rows took {stopwatch.elapsedTime() / 60:.2f} minutes")


if __name__ == "__main__":
    main()


"""

diagnosis.csv
         subject_id  chart_time icd10_cm
0         190852492      325440      R06
1         190852492      325440      G20
2         142367193           0      I61
3         178346414       80640      A16
4         178346414       80640      J98
...             ...         ...      ...
2464615   195448932      172800      H02
2464616   195448932      172800      L90
2464617   195448932      172800      S04
2464618   155658270       24480      K81
2464619   155658270       34560      K81

[2464620 rows x 3 columns]
File ./dataset/diagnosis.csv, (rows, cols)=(2464620, 3), took 0.01 minutes




department.csv
   Abbreviations                Full name
0             AN           Anesthesiology
1            CTS  Cardio-Thoracic Surgery
2             DM              Dermatology
3             EM       Emergency Medicine
4             GS          General Surgery
5             IM        Internal Medicine
6             NS             Neurosurgery
7             OG  Obstetrics & Gynecology
8             OL          Oto-laryngology
9             OS       Orthopedic Surgery
10            OT            Ophthalmology
11           PED               Pediatrics
12            PS          Plastic Surgery
13           RAD                Radiology
14            RO       Radiation Oncology
15            UR                  Urology
File ./dataset/department.csv, (rows, cols)=(16, 2), took 0.00 minutes

Conversation with Julius, really only care about these surgeries/departments
Abbreviations,Full name
CTS,Cardio-Thoracic Surgery
GS,General Surgery
NS,Neurosurgery
OG,Obstetrics & Gynecology
OL,Otolaryngology
OS,Orthopedic Surgery
OT,Ophthalmology
PS,Plastic Surgery
UR,Urology


labs.csv
          subject_id  chart_time      item_name  value
0          133338290       86155  total_protein    6.9
1          133338290       86155         sodium  140.0
2          133338290       86155      potassium    4.4
3          133338290       86155       platelet  152.0
4          133338290       93150        glucose  120.0
...              ...         ...            ...    ...
19503330   155658270       35270             hb   15.0
19503331   155658270       35270            hct   44.1
19503332   155658270       35270       platelet  245.0
19503333   155658270       35270            seg   79.9
19503334   155658270       35270     lymphocyte   11.8

[19503335 rows x 4 columns]
File ./dataset/labs.csv, (rows, cols)=(19503335, 4), took 0.06 minutes
operations.csv
op_id
subject_id
hadm_id
case_id
opdate
age
sex
weight
height
race
asa
emop
department
antype
icd10_pcs
orin_time
orout_time
opstart_time
opend_time
admission_time
discharge_time
anstart_time
anend_time
cpbon_time
cpboff_time
icuin_time
icuout_time
inhosp_death_time
allcause_death_time
            op_id  subject_id    hadm_id  case_id  ...  icuin_time  icuout_time inhosp_death_time  allcause_death_time
0       484069807   178742874  229842382      NaN  ...         NaN          NaN               NaN                  NaN
1       446270725   158995752  257857903      NaN  ...      1550.0      19595.0           69860.0             106560.0
2       406892271   108553242  200664328      NaN  ...         NaN          NaN               NaN             718560.0
3       478413008   133278262  277235295      NaN  ...         NaN          NaN               NaN                  NaN
4       468516791   116924034  299190423      NaN  ...         NaN          NaN               NaN                  NaN
...           ...         ...        ...      ...  ...         ...          ...               ...                  ...
130955  449124488   138484174  228449654      NaN  ...         NaN          NaN               NaN                  NaN
130956  461252752   126772283  273139806      NaN  ...         NaN          NaN               NaN                  NaN
130957  471834474   144363433  275833861      NaN  ...         NaN          NaN               NaN                  NaN
130958  419787421   195835964  293939099      NaN  ...     13955.0      15120.0               NaN             613440.0
130959  493136728   148443464  280269654      NaN  ...         NaN          NaN               NaN                  NaN

[130960 rows x 29 columns]
File ./dataset/operations.csv, (rows, cols)=(130960, 29), took 0.00 minutes




vitals.csv
              op_id  subject_id  chart_time item_name  value
0         435959808   181409183        1985    minvol    4.4
1         435959808   181409183        1985        vt  512.0
2         435959808   181409183        1985        rr    9.0
3         435959808   181409183        1985       pip   23.0
4         435959808   181409183        2005    minvol    4.4
...             ...         ...         ...       ...    ...
66127935  447098707   159399111     1511530  nibp_dbp   92.0
66127936  447098707   159399111     1511530  nibp_mbp  119.0
66127937  447098707   159399111     1511530      spo2  100.0
66127938  447098707   159399111     1511535        hr   78.0
66127939  447098707   159399111     1511535      spo2  100.0

[66127940 rows x 5 columns]
File ./dataset/vitals.csv, (rows, cols)=(66127940, 5), took 0.24 minutes


anomaly detection

ward_vitals.csv
          subject_id  chart_time item_name  value
0          104192463         580      spo2   98.0
1          104192463         580  nibp_sbp  169.0
2          104192463         580        hr   96.0
3          104192463         580  nibp_dbp  100.0
4          104192463         580        bt   36.4
...              ...         ...       ...    ...
45796479   155523991        6465      iabp    1.0
45796480   155523991        6480      iabp    1.0
45796481   155523991        6540      iabp    1.0
45796482   155523991        6570      iabp    1.0
45796483   155523991        6720      iabp    1.0

[45796484 rows x 4 columns]
File ./dataset/ward_vitals.csv, (rows, cols)=(45796484, 4), took 0.14 minutes




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

"""
