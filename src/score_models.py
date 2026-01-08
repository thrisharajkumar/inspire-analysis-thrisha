

def compute_news2_score(resp_rate, spo2, temp, systolic_bp, heart_rate, consciousness, oxygen_support):
    """
    Calculate the National Early Warning Score (NEWS2) based on vital signs.

    Parameters:
    - resp_rate: Respiratory rate (breaths per minute)
    - spo2: Oxygen saturation (%)
    - temp: Temperature (°C)
    - systolic_bp: Systolic blood pressure (mmHg)
    - heart_rate: Heart rate (beats per minute)
    - consciousness: Level of consciousness ('A', 'C', 'V', 'P', 'U')
    - oxygen_support: Boolean indicating if patient is on supplemental oxygen

    Returns:
    - Total NEWS2 score
    """
    # https://www.freeonlinecalc.com/national-early-warning-score-news2-calculator.html
    # I need test data to ensure a "National Early Warning Score (NEWS)" function is correct.
    # The test data should have (resp_rate, spo2, temp, systolic_bp, heart_rate,
    # consciousness, oxygen_support) inputs and the correct NEWS score for those inputs.

    score = 0

    # Respiratory rate
    if resp_rate <= 8:
        score += 3
    elif 9 <= resp_rate <= 11:
        score += 1
    elif 12 <= resp_rate <= 20:
        score += 0
    elif 21 <= resp_rate <= 24:
        score += 2
    elif resp_rate >= 25:
        score += 3

    # Oxygen saturation (SpO2) - scale 1 used here (for general patients)
    if spo2 <= 91:
        score += 3
    elif 92 <= spo2 <= 93:
        score += 2
    elif 94 <= spo2 <= 95:
        score += 1
    elif spo2 >= 96:
        score += 0

    # Supplemental oxygen
    if oxygen_support:
        score += 2

    # Temperature
    if temp <= 35.0:
        score += 3
    elif 35.1 <= temp <= 36.0:
        score += 1
    elif 36.1 <= temp <= 38.0:
        score += 0
    elif 38.1 <= temp <= 39.0:
        score += 1
    elif temp >= 39.1:
        score += 2

    # Systolic blood pressure
    if systolic_bp <= 90:
        score += 3
    elif 91 <= systolic_bp <= 100:
        score += 2
    elif 101 <= systolic_bp <= 110:
        score += 1
    elif 111 <= systolic_bp <= 219:
        score += 0
    elif systolic_bp >= 220:
        score += 3

    # Heart rate
    if heart_rate <= 40:
        score += 3
    elif 41 <= heart_rate <= 50:
        score += 1
    elif 51 <= heart_rate <= 90:
        score += 0
    elif 91 <= heart_rate <= 110:
        score += 1
    elif 111 <= heart_rate <= 130:
        score += 2
    elif heart_rate >= 131:
        score += 3

    # Level of consciousness (AVPU/ACVPU)
    if consciousness.upper() in ['C', 'V', 'P', 'U']:
        score += 3

    return score


"""
Reads all packets from pcap into memory.
"""
def main():
    score0 = compute_news2_score(
        resp_rate=22,
        spo2=93,
        temp=37.2,
        systolic_bp=105,
        heart_rate=95,
        consciousness='A',
        oxygen_support=True
    )
    print("NEWS2 Score0:", score0)

    score1 = compute_news2_score(
        resp_rate=25,
        spo2=93,
        temp=37.5,
        systolic_bp=130,
        heart_rate=100,
        consciousness='A',
        oxygen_support=False
    )
    print("NEWS2 Score1:", score1)

    score2 = compute_news2_score(
        resp_rate=15,
        spo2=95,
        temp=36.5,
        systolic_bp=110,
        heart_rate=80,
        consciousness='C',
        oxygen_support=True
    )
    print("NEWS2 Score2:", score2)

    score3 = compute_news2_score(
        resp_rate=30,
        spo2=88,
        temp=38.0,
        systolic_bp=90,
        heart_rate=120,
        consciousness='P',
        oxygen_support=True
    )
    print("NEWS2 Score3:", score3)

if __name__ == "__main__":
    main()


"""
Ground truth derived from https://www.freeonlinecalc.com/national-early-warning-score-news2-calculator.html
Test Data 1: (resp_rate: 25, spo2: 92, temp: 37.5, systolic_bp: 130, heart_rate: 100, consciousness: Alert, oxygen_support: No) - NEWS Score: 6
Test Data 2: (resp_rate: 15, spo2: 95, temp: 36.5, systolic_bp: 110, heart_rate: 80, consciousness: Confused, oxygen_support: Yes) - NEWS Score: 7
Test Data 3: (resp_rate: 30, spo2: 88, temp: 38.0, systolic_bp: 90, heart_rate: 120, consciousness: Pains, oxygen_support: Yes) - NEWS Score: 10
"""
