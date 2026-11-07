"""Net technology abatement columns for the fleet-model result sheets (pm*).

Under the avoided accounting the three technology contributions add up to the reported total only after
two deductions:
  CC: captured CO2 minus the uncaptured CO2 of the extra coal burnt for the capture energy
      (cc_net_abatement_ton, written by the model);
  BC: biomass CO2 benefit minus the biomass supply-chain and haulage emissions (logistics_emission_ton).
  ec_abatement_ton + bc_net_abatement_ton + cc_net_abatement_ton = total_abatement_ton.
Under the captured accounting both deductions are zero and the net columns equal the gross ones.
"""

import pandas as pd


def add_net_abatement_columns(pdf: pd.DataFrame) -> pd.DataFrame:
    if pdf.empty:
        return pdf
    pdf = pdf.copy()

    def col(name: str) -> pd.Series:
        if name in pdf.columns:
            return pd.to_numeric(pdf[name], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=pdf.index)

    if "bc_abatement_ton" in pdf.columns:
        pdf["bc_net_abatement_ton"] = col("bc_abatement_ton") - col("logistics_emission_ton")
    if "cc_net_abatement_ton" not in pdf.columns and "cc_capture_ton" in pdf.columns:
        pdf["cc_net_abatement_ton"] = col("cc_capture_ton")
    return pdf
