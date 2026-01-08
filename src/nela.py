# Derived from "Technical Document NELA_PRS_Overview_Coefficients April 2023.pdf"

import math

def compute_nela_score(
        Age_cent,
        ASA_3=0, ASA_4=0, ASA_5=0,
        Albumin=0,
        Pulse_cent=0, Pulse_cent2=0,
        SystolicBP_cent=0, SystolicBP_cent2=0,
        LN_Urea_cent=0,
        LN_WBC_cent=0, LN_WBC_cent2=0,
        GCS_14=0, GCS_3_13=0,
        Malignancy_Primary=0,
        Malignancy_Nodal=0,
        Malignancy_Distant=0,
        Respiratory_2=0, Respiratory_3=0,
        Urgency_6_18=0,
        Urgency_2_6=0,
        Urgency_lt_2=0,
        Indication_Sepsis=0,
        Indication_Ischaemia=0,
        Indication_Bleeding=0,
        Soiling_Severe=0
):
    logit = (
            -3.04678
            + 0.06660 * Age_cent
            + 1.13007 * ASA_3 + 1.76293 * ASA_4 + 2.55345 * ASA_5
            - 0.03021 * ASA_3 * Age_cent
            - 0.03356 * ASA_4 * Age_cent
            - 0.04676 * ASA_5 * Age_cent
            - 0.04323 * Albumin
            + 0.01265 * Pulse_cent
            - 0.00012 * Pulse_cent2
            - 0.00683 * SystolicBP_cent
            + 0.00011 * SystolicBP_cent2
            + 0.38002 * LN_Urea_cent
            + 0.02041 * LN_WBC_cent
            + 0.24153 * LN_WBC_cent2
            + 0.41557 * GCS_14
            + 0.64480 * GCS_3_13
            + 0.19201 * Malignancy_Primary
            + 0.50610 * Malignancy_Nodal
            + 0.94309 * Malignancy_Distant
            + 0.35378 * Respiratory_2
            + 0.60700 * Respiratory_3
            + 0.03782 * Urgency_6_18
            + 0.14779 * Urgency_2_6
            + 0.57310 * Urgency_lt_2
            + 0.02812 * Indication_Sepsis
            + 0.56948 * Indication_Ischaemia
            - 0.40615 * Indication_Bleeding
            + 0.29453 * Soiling_Severe
    )

    probability = 1 / (1 + math.exp(-logit))
    return probability

def main():
    risk = compute_nela_score(
        Age_cent=5,
        ASA_3=1,
        Albumin=35,
        Pulse_cent=10, Pulse_cent2=100,
        SystolicBP_cent=-5, SystolicBP_cent2=25,
        LN_Urea_cent=0.3,
        LN_WBC_cent=0.2, LN_WBC_cent2=0.04,
        GCS_14=1,
        Malignancy_Primary=1,
        Respiratory_2=1,
        Urgency_2_6=1,
        Indication_Sepsis=1,
        Soiling_Severe=1
    )
    print(f"NELA predicted risk: {risk:.4f}")


if __name__ == "__main__":
    main()