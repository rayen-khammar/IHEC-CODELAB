import pandas as pd

def clean_number(x):
    if pd.isna(x):
        return None

    x = str(x).strip()

    if x in ["—", "-", "", "None", "null"]:
        return None

    x = x.replace(" ", "")
    x = x.replace(",", ".")
    x = x.replace("%", "")

    try:
        return float(x)
    except:
        return None


def clean_text(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    if x in ["—", "-", ""]:
        return None
    return x
