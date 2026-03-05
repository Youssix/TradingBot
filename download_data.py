import MetaTrader5 as mt5
import pandas as pd

if not mt5.initialize():
    print(f"MT5 initialize failed: {mt5.last_error()}")
    quit()

mt5.symbol_select("XAUUSD", True)
print(f"Account: {mt5.account_info().login}, Server: {mt5.account_info().server}\n")

CHUNK = 50000

def download_all(timeframe_name, timeframe):
    all_frames = []
    offset = 0
    while True:
        rates = mt5.copy_rates_from_pos("XAUUSD", timeframe, offset, CHUNK)
        if rates is None or len(rates) == 0:
            break
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        print(f"  {timeframe_name} offset {offset}: {len(df)} bars ({df['time'].iloc[0]} to {df['time'].iloc[-1]})")
        all_frames.append(df)
        if len(rates) < CHUNK:
            break
        offset += CHUNK

    if not all_frames:
        print(f"  No {timeframe_name} data available.")
        return None

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    return combined

def save_by_year(df, timeframe_name):
    df["year"] = df["time"].dt.year
    for year, group in df.groupby("year"):
        group = group.drop(columns=["year"])
        filename = f"xauusd_{timeframe_name}_{year}.csv"
        group.to_csv(filename, index=False)
        print(f"  {filename}: {len(group)} bars ({group['time'].iloc[0].date()} to {group['time'].iloc[-1].date()})")

# Download H1
print("=== Downloading H1 ===")
h1 = download_all("H1", mt5.TIMEFRAME_H1)
if h1 is not None:
    print(f"\nH1 total: {len(h1)} bars ({h1['time'].iloc[0].date()} to {h1['time'].iloc[-1].date()})")
    print("Saving by year:")
    save_by_year(h1, "h1")

# Download M1
print("\n=== Downloading M1 ===")
m1 = download_all("M1", mt5.TIMEFRAME_M1)
if m1 is not None:
    print(f"\nM1 total: {len(m1)} bars ({m1['time'].iloc[0].date()} to {m1['time'].iloc[-1].date()})")
    print("Saving by year:")
    save_by_year(m1, "m1")

mt5.shutdown()
print("\nDone!")
