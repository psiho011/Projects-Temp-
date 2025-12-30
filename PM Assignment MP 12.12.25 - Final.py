import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path


DATA_DIR = Path(
    r"C:\Users\mpsih\Dropbox\Masters Program School Work\Carlson Funds Enterprise\4 - Documents & Assignments\2 - PM Assignment")

PORT_FILE = DATA_DIR / "25_Portfolios_5x5.csv"
FF_FILE   = DATA_DIR / "F-F_Research_Data_Factors.csv"
Q5_FILE   = DATA_DIR / "q5_factors_monthly_2024.csv"

START_YM = 193107  # per assignment


def ym_to_period(ym: int) -> pd.Period:
    y = ym // 100
    m = ym % 100
    return pd.Period(f"{y}-{m:02d}", freq="M")

def read_kf_section_monthly_25p(path: Path) -> pd.DataFrame:
    """
    Reads the FIRST 'Average Value Weighted Returns -- Monthly' """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Find header row that begins with ",SMALL ..."
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith(",SMALL"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find the  header row in portfolio CSV.")

    header = lines[header_idx].split(",")
    # header[0] is blank (date column)
    col_names = [h.strip() for h in header[1:]]

    data_rows = []
    for line in lines[header_idx + 1:]:
        s = line.strip()
        if not s:
            break

        # stop if we hit a non-date row
        if not re.match(r"^\d{6},", s):
            break

        parts = s.split(",")
        ym = int(parts[0])
        vals = parts[1:1+len(col_names)]
        if len(vals) != len(col_names):
           
            continue

        data_rows.append([ym] + vals)

    df = pd.DataFrame(data_rows, columns=["ym"] + col_names)

    # numeric cleanup
    for c in col_names:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c].isin([-99.99, -999.0, -999.99]), c] = np.nan

    df["date"] = df["ym"].apply(ym_to_period)
    df = df.drop(columns=["ym"]).set_index("date").sort_index()
    return df

def read_ff_factors(path: Path) -> pd.DataFrame:
    """
    Reads Fama-French 3 factors monthly CSV (Ken French style)."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Find the header line that contains Mkt-RF
    header_idx = None
    for i, line in enumerate(lines):
        if "Mkt-RF" in line and "SMB" in line and "HML" in line and "RF" in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find FF factor header row.")

    # Data starts next line, ends when lines stop looking like YYYYMM,...
    data = []
    for line in lines[header_idx + 1:]:
        s = line.strip()
        if not re.match(r"^\d{6},", s):
            continue
        parts = s.split(",")
        ym = int(parts[0])
        if len(parts) < 5:
            continue
        data.append([ym] + parts[1:5])

    df = pd.DataFrame(data, columns=["ym", "Mkt-RF", "SMB", "HML", "RF"])
    for c in ["Mkt-RF", "SMB", "HML", "RF"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["date"] = df["ym"].apply(ym_to_period)
    df = df.drop(columns=["ym"]).set_index("date").sort_index()
    return df

def run_ts_regressions(Y: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for asset in Y.columns:
        y = Y[asset]
        df = pd.concat([y, X], axis=1, join="inner").dropna()
        if df.shape[0] < 24:
            # too few observations
            continue

        yv = df[asset]
        Xv = sm.add_constant(df[X.columns])

        res = sm.OLS(yv, Xv).fit()

        row = {
            "asset": asset,
            "avg_excess": yv.mean(),
            "alpha": res.params.get("const", np.nan),
            "t_alpha": res.tvalues.get("const", np.nan),
            "nobs": int(res.nobs),
            "r2": res.rsquared,
        }
        for k in X.columns:
            row[f"beta_{k}"] = res.params.get(k, np.nan)
        out_rows.append(row)

    return pd.DataFrame(out_rows).set_index("asset").sort_index()

def ff_5x5_table(series_by_asset: pd.Series) -> pd.DataFrame:
    # Expected order from the file:
    rows = ["SMALL", "ME2", "ME3", "ME4", "BIG"]
    cols = ["LoBM", "BM2", "BM3", "BM4", "HiBM"]

    table = pd.DataFrame(index=rows, columns=cols, dtype=float)

    for asset, val in series_by_asset.items():
        a = asset.strip()

        # First row uses "SMALL LoBM" / "SMALL HiBM"
        if a.startswith("SMALL "):
            c = a.replace("SMALL ", "").strip()
            table.loc["SMALL", c] = val
            continue

        # Last row uses "BIG LoBM" / "BIG HiBM"
        if a.startswith("BIG "):
            c = a.replace("BIG ", "").strip()
            table.loc["BIG", c] = val
            continue

        # Middle rows are "ME{n} BM{k}"
        m = re.match(r"^(ME[2-4])\s+(BM[2-4])$", a)
        if m:
            r, c = m.group(1), m.group(2)
            table.loc[r, c] = val

    return table


ports = read_kf_section_monthly_25p(PORT_FILE)        # 25 portfolios, percent
ff3   = read_ff_factors(FF_FILE)                      # Mkt-RF, SMB, HML, RF, percent
q5    = pd.read_csv(Q5_FILE)                          # already clean in your upload

# Make q5 monthly Period index
q5["date"] = q5.apply(lambda r: pd.Period(f"{int(r['year'])}-{int(r['month']):02d}", freq="M"), axis=1)
q5 = q5.set_index("date").sort_index()

# Apply assignment trims

ports = ports.loc[ports.index >= ym_to_period(START_YM)]
ff3   = ff3.loc[ff3.index >= ym_to_period(START_YM)]

# "chop off last three months of the factors"
ff3 = ff3.iloc[:-3, :]

# Align on common dates
common = ports.index.intersection(ff3.index)
ports = ports.loc[common]
ff3   = ff3.loc[common]

# Excess returns for portfolios 
ports_excess = ports.sub(ff3["RF"], axis=0)


# 1) CAPM regressions: y = alpha + beta*(Mkt-RF)

capm_X = ff3[["Mkt-RF"]]
capm_res = run_ts_regressions(ports_excess, capm_X)


# 2) FF3 regressions: y = alpha + b*(Mkt-RF) + s*SMB + h*HML

ff3_X = ff3[["Mkt-RF", "SMB", "HML"]]
ff3_res = run_ts_regressions(ports_excess, ff3_X)


# 3) q-factor model (starts 1964 per assignment)
q5 = q5.loc[q5.index >= pd.Period("1964-01", freq="M")]
ports_q = ports.loc[ports.index.intersection(q5.index)]
q5 = q5.loc[ports_q.index]

ports_q_excess = ports_q.sub(q5["R_F"], axis=0)

q5_X = pd.DataFrame(
    {
        "MKT": q5["R_MKT"] - q5["R_F"],
        "ME":  q5["R_ME"],
        "IA":  q5["R_IA"],
        "ROE": q5["R_ROE"],
    },
    index=q5.index,
)

q5_res = run_ts_regressions(ports_q_excess, q5_X)


OUT = DATA_DIR / "outputs"
OUT.mkdir(exist_ok=True)

# --- keep CSV outputs (optional but nice) ---
capm_res.to_csv(OUT / "capm_results_long.csv")
ff3_res.to_csv(OUT / "ff3_results_long.csv")
q5_res.to_csv(OUT / "q5_results_long.csv")

def export_tables(res: pd.DataFrame, prefix: str):
    tables = {
        "avg_excess": ff_5x5_table(res["avg_excess"]),
        "alpha":      ff_5x5_table(res["alpha"]),
        "t_alpha":    ff_5x5_table(res["t_alpha"]),
        "r2":         ff_5x5_table(res["r2"]),
    }
    for c in res.columns:
        if c.startswith("beta_"):
            tables[c] = ff_5x5_table(res[c])

    # also keep per-table CSVs if you want
    for name, tbl in tables.items():
        tbl.to_csv(OUT / f"{prefix}_{name}_5x5.csv")

    return tables

capm_tables = export_tables(capm_res, "capm")
ff3_tables  = export_tables(ff3_res,  "ff3")
q5_tables   = export_tables(q5_res,   "q5")
out_xlsx = OUT / "assignment_outputs.xlsx"

with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    # long-form
    capm_res.to_excel(writer, sheet_name="CAPM_long")
    ff3_res.to_excel(writer, sheet_name="FF3_long")
    q5_res.to_excel(writer, sheet_name="Q5_long")

    # 5x5 tables: CAPM
    for name, tbl in capm_tables.items():
        sheet = f"CAPM_{name}"[:31]
        tbl.to_excel(writer, sheet_name=sheet)

    # 5x5 tables: FF3
    for name, tbl in ff3_tables.items():
        sheet = f"FF3_{name}"[:31]
        tbl.to_excel(writer, sheet_name=sheet)

    # 5x5 tables: Q5
    for name, tbl in q5_tables.items():
        sheet = f"Q5_{name}"[:31]
        tbl.to_excel(writer, sheet_name=sheet)

print("Done. Wrote outputs to:", OUT.resolve())
print("Excel workbook saved to:", out_xlsx.resolve())
