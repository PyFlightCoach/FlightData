from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class BinData:
    dfs: dict[str, pd.DataFrame]

    def __getattr__(self, name):
        if name in self.dfs:
            return self.dfs[name]
        raise AttributeError(f"No such attribute: {name}")

    def get_dft(self, df_name: str, t: float, coren: str|None='C', coreid: int=0) -> pd.DataFrame:
        bdx = self.dfs[df_name].copy()
        bdx['TimeUS'] = bdx.TimeUS / 1e6
        if coren and coren in bdx.columns:
            bdx = bdx.loc[bdx[coren]==coreid]
        bdx = bdx.set_index('TimeUS')
        return bdx.iloc[bdx.index.get_indexer([t], method='nearest')]

    def parameters(self) -> dict[str, pd.DataFrame]:
        gb = self.PARM.groupby("Name")

        parms = {}
        for gn in gb.groups.keys():
            gr = gb.get_group(gn)
            parms[gn] = gr.loc[
                abs(gr.Value.diff().fillna(1)) > 0, ["timestamp", "TimeUS", "Value"]
            ].set_index("timestamp")
        return parms

    @staticmethod
    def _gpsTimeToTime(week, msec):
        """convert GPS week and TOW to a time in seconds since 1970"""
        epoch = 86400 * (10 * 365 + int((1980 - 1969) / 4) + 1 + 6 - 2)
        return epoch + 86400 * 7 * week + msec * 0.001 - 18

    @staticmethod
    def parse_json(bindata: dict[str, dict[str, list]]) -> BinData:
        # dfs = {k: pd.DataFrame(v) for k,v in bindata.items()}

        dfs: dict[str, pd.DataFrame] = {}
        groups = {}
        for k, v in bindata.items():
            if k == "PARM":
                new_df = pd.DataFrame(
                    [
                        pd.Series(v["time_boot_s"]).reset_index(drop=True),
                        pd.Series(v["Name"]),
                        pd.Series(v["Value"]).reset_index(drop=True),
                    ],
                    index=["time_boot_s", "Name", "Value"],
                ).T
                if "Default" in v:
                    new_df["Default"] = v["Default"]
            elif k == "MSG":
                new_df = pd.DataFrame(
                    [
                        pd.Series(v["time_boot_s"]).reset_index(drop=True),
                        pd.Series(v["Message"]),
                    ],
                    index=["time_boot_s", "Message"],
                ).T
            else:
                try:
                    new_df = pd.DataFrame(v)
                except Exception as ex:
                    logger.info(f"Error parsing {k}: {ex}")
                    new_df = None

            if new_df is not None:
                if "[" not in k:
                    dfs[k] = new_df
                else:
                    nk = k.split("[")[0]
                    if nk not in groups:
                        groups[nk] = []
                    groups[nk].append(new_df)

        for k, v in groups.items():
            dfs[k] = pd.concat(v).sort_values("time_boot_s")

        if "GPS" in dfs:
            start_time = (
                BinData._gpsTimeToTime(dfs["GPS"].GWk.iloc[0], dfs["GPS"].GMS.iloc[0])
                - dfs["GPS"].time_boot_s.iloc[0]
            )
        else:
            start_time = 0

        def process_df(df: pd.DataFrame) -> pd.DataFrame:
            df.insert(0, "TimeUS", np.floor(df.time_boot_s * 1e6))  # ms
            df.insert(0, "timestamp", start_time + df.time_boot_s)

            return df.drop(columns="time_boot_s")

        dfs = {k: process_df(v) for k, v in dfs.items() if not v.empty}

        return BinData(dfs)
