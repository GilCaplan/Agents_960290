import gc
import json
import pandas as pd
import os
import re


class WeatherService:
    def __init__(self, data_dir="city_data"):
        self.data_dir = data_dir
        self._daily_cache = {}  # city_key -> compact daily DataFrame (one row per day)
        if os.path.exists(data_dir):
            self._available = {f.replace(".json", "").lower(): f
                               for f in os.listdir(data_dir) if f.endswith('.json')}
            print(f"--- Weather Service ready: {len(self._available)} cities indexed ---")
        else:
            self._available = {}
            print(f"WARNING: Directory {data_dir} not found.")

    def _get_daily(self, city_key: str) -> pd.DataFrame | None:
        """
        Load city JSON once, aggregate to one row per day, cache the compact result.
        First call: ~1-2s (loads + aggregates 13-16MB file).
        All subsequent calls for same city: <1ms from cache.
        """
        if city_key in self._daily_cache:
            return self._daily_cache[city_key]

        filename = self._available.get(city_key)
        if not filename:
            return None

        try:
            file_path = os.path.join(self.data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)

            df = pd.DataFrame(data)
            del data
            gc.collect()

            df['date'] = pd.to_datetime(df['date'], dayfirst=True).dt.tz_localize(None)
            df['_day'] = df['date'].dt.normalize()

            # Coerce all non-date columns to numeric (converts '-' / garbage → NaN)
            for c in df.columns:
                if c not in ('date', '_day', 'sname'):
                    df[c] = pd.to_numeric(df[c], errors='coerce')

            # Build named aggregation dynamically from available numeric columns
            agg = {}
            cols = set(df.columns)

            if 'TD'   in cols: agg.update({'TD_avg': ('TD', 'mean'), 'TD_max': ('TD', 'max'), 'TD_min': ('TD', 'min')})
            # TG = ground/grass temperature. Some stations (e.g. Beer Sheva) report
            # ONLY TG and never air temperature (TD); keep it as a labelled fallback.
            if 'TG'   in cols: agg.update({'TG_avg': ('TG', 'mean'), 'TG_max': ('TG', 'max'), 'TG_min': ('TG', 'min')})
            if 'RH'   in cols: agg['RH_avg']       = ('RH',   'mean')
            # min_count=1 → stays NaN (→ "N/A") if the station never reports rain,
            # instead of a misleading 0.0 mm.
            if 'Rain' in cols: agg['Rain_total']    = ('Rain', lambda s: s.sum(min_count=1))
            if 'WS'   in cols: agg.update({'WS_avg': ('WS', 'mean'), 'WS_max': ('WS', 'max')})
            if 'WD'   in cols: agg['WD_avg']        = ('WD',   'mean')
            if 'STDwd' in cols: agg['STDwd_avg']    = ('STDwd','mean')

            # Any remaining numeric cols → daily mean
            skip = {'TD', 'RH', 'Rain', 'WS', 'WD', 'STDwd', 'Time'}
            for c in cols - skip:
                if c.startswith('_') or c in ('date', 'sname'):
                    continue
                if pd.api.types.is_numeric_dtype(df[c]) and f"{c}_avg" not in agg:
                    agg[f"{c}_avg"] = (c, 'mean')

            daily = df.groupby('_day').agg(**agg).round(2).reset_index()
            daily.rename(columns={'_day': 'date'}, inplace=True)

            del df
            gc.collect()

            self._daily_cache[city_key] = daily
            print(f"[Weather] {city_key}: cached {len(daily)} days")
            return daily

        except Exception as e:
            print(f"[Weather] Error loading {filename}: {e}")
            return None

    # ------------------------------------------------------------------
    def get_weather(self, query: str) -> str:
        """
        Parse city + date from agent query string, return rich context:
        today's conditions + 7-day + 30-day summaries for agricultural planning.
        """
        print(f"\n[DEBUG] Weather Tool Called -> Query: {query}")

        # Extract date
        date_match = re.search(r'\d{4}-\d{2}-\d{2}', query)
        date_str = date_match.group(0) if date_match else None

        # Extract city
        city = query.replace(date_str, "").replace(" on ", "").strip() if date_str else query.strip()

        if "{" in city:
            import ast
            try:
                parsed = ast.literal_eval(city)
                if isinstance(parsed, dict):
                    city = list(parsed.values())[0]
            except Exception:
                pass

        user_query = city.lower().strip()

        # Hebrew and common-variant aliases → canonical file key
        CITY_ALIASES = {
            # Eilat
            "אילת": "eilat",
            # Ariel
            "אריאל": "ariel",
            # Beer Sheva
            "באר שבע": "beer_sheva",
            "beer sheva": "beer_sheva",
            "beersheba": "beer_sheva",
            "be'er sheva": "beer_sheva",
            # Ashdod
            "אשדוד": "ashdod",
            # Ashkelon
            "אשקלון": "ashkelon",
            # Hadera
            "חדרה": "hadera",
            # Haifa Technion
            "חיפה": "haifa_technion",
            "חיפה טכניון": "haifa_technion",
            "haifa": "haifa_technion",
            "haifa technion": "haifa_technion",
            # Haifa Bate Zakuk
            "חיפה בתי זיקוק": "haifa_bate_zakuk",
            "haifa bate zakuk": "haifa_bate_zakuk",
            # Jerusalem
            "ירושלים": "jerusalem_center",
            "ירושלים מרכז": "jerusalem_center",
            "jerusalem": "jerusalem_center",
            "jerusalem center": "jerusalem_center",
            # Nitzan
            "ניצן": "nitzan",
            # Yotvata
            "יוטבתה": "yotvata",
            "יטבתה": "yotvata",
            # Avne Eitan
            "אבני איתן": "avne_eitan",
            "avne eitan": "avne_eitan",
            # Lev Kineret
            "לב כנרת": "lev_kineret",
            "כנרת": "lev_kineret",
            "lev kineret": "lev_kineret",
            "kineret": "lev_kineret",
            # Maale Gilboa
            "מעלה גלבוע": "maale_gilboa",
            "גלבוע": "maale_gilboa",
            "maale gilboa": "maale_gilboa",
            # Tel Aviv Beach
            "תל אביב": "tlv_beach",
            "חוף תל אביב": "tlv_beach",
            "tel aviv": "tlv_beach",
            "tlv": "tlv_beach",
            "tlv beach": "tlv_beach",
            # Zichron Yaakov
            "זכרון יעקב": "zichron_yaakov",
            "זיכרון יעקב": "zichron_yaakov",
            "zichron yaakov": "zichron_yaakov",
            "zikhron yaakov": "zichron_yaakov",
        }

        # 1. Exact alias lookup (Hebrew or known variants)
        matched_key = CITY_ALIASES.get(user_query)

        # 2. Prefix/partial alias lookup (e.g. "חיפה טכניון" when query is "חיפה ...")
        if matched_key is None:
            for alias, key in CITY_ALIASES.items():
                if alias in user_query or user_query in alias:
                    matched_key = key
                    break

        # 3. Match against file keys (sorted for determinism)
        if matched_key is None:
            for file_key in sorted(self._available):
                clean = file_key.replace("_", " ")
                if user_query == clean or clean in user_query or user_query in clean:
                    matched_key = file_key
                    break

        if matched_key is None:
            return f"Error: City '{city}' not found. Available cities: {', '.join(sorted(self._available))}"

        daily = self._get_daily(matched_key)
        if daily is None or daily.empty:
            return "Error: Could not load weather data."

        # Target date
        target = pd.to_datetime(date_str) if date_str else daily['date'].iloc[-1]

        # Find closest available day (sub-millisecond on cached daily DataFrame)
        idx = (daily['date'] - target).abs().idxmin()
        today = daily.loc[idx]
        actual_date = today['date'].strftime('%Y-%m-%d')
        print(f"[Weather] {matched_key}: requested {date_str} → matched {actual_date}")

        # 7-day and 30-day windows (days strictly before target)
        prev7  = daily[(daily['date'] >= target - pd.Timedelta(days=7))  & (daily['date'] < target)]
        prev30 = daily[(daily['date'] >= target - pd.Timedelta(days=30)) & (daily['date'] < target)]

        def _has(row, key):
            v = row.get(key)
            return not (v is None or (isinstance(v, float) and pd.isna(v)))

        def _val(row, key, unit=""):
            if not _has(row, key):
                return "N/A"
            return f"{row.get(key)}{unit}"

        # Temperature line: prefer air temp (TD); if the station only reports ground
        # temperature (TG) — as Beer Sheva does — fall back to it with a clear label
        # instead of showing N/A across the board.
        if _has(today, 'TD_avg'):
            temp_line = (f"  Air temperature : avg {_val(today,'TD_avg','°C')}  "
                         f"max {_val(today,'TD_max','°C')}  min {_val(today,'TD_min','°C')}")
        elif _has(today, 'TG_avg'):
            temp_line = (f"  Ground temperature (station has no air-temp sensor) : "
                         f"avg {_val(today,'TG_avg','°C')}  max {_val(today,'TG_max','°C')}  "
                         f"min {_val(today,'TG_min','°C')}")
        else:
            temp_line = "  Temperature : N/A (not reported by this station)"

        lines = [
            "REAL MEASURED WEATHER DATA (treat this reference date as 'today' for advice):",
            f"Location: {matched_key}  |  Reference date: {actual_date}",
            "",
            "── TODAY'S CONDITIONS ──────────────────────────────",
            temp_line,
            f"  Humidity    : {_val(today,'RH_avg','%')}",
            f"  Rainfall    : {_val(today,'Rain_total',' mm')}",
            f"  Wind        : avg {_val(today,'WS_avg',' m/s')}  max {_val(today,'WS_max',' m/s')}  direction {_val(today,'WD_avg','°')}",
        ]

        def _temp_summary(window):
            """Avg/max/min temp line, falling back to ground temp (TG) if no air temp."""
            if 'TD_avg' in window.columns and window['TD_avg'].notna().any():
                return (f"  Avg air temp: {window['TD_avg'].mean():.1f}°C  "
                        f"(max {window['TD_max'].max():.1f}°C  min {window['TD_min'].min():.1f}°C)")
            if 'TG_avg' in window.columns and window['TG_avg'].notna().any():
                return (f"  Avg ground temp: {window['TG_avg'].mean():.1f}°C  "
                        f"(max {window['TG_max'].max():.1f}°C  min {window['TG_min'].min():.1f}°C)  [no air-temp sensor]")
            return ""

        def _frost(window):
            col = 'TD_min' if 'TD_min' in window.columns else ('TG_min' if 'TG_min' in window.columns else None)
            return int((window[col] < 0).sum()) if col else 'N/A'

        def _mean_pct(window, col):
            if col in window.columns and window[col].notna().any():
                return f"  Avg humidity: {window[col].mean():.0f}%"
            return ""

        def _rain(window):
            if 'Rain_total' in window.columns and window['Rain_total'].notna().any():
                return f"  Total rain  : {window['Rain_total'].sum():.1f} mm"
            return ""

        if len(prev7) > 0:
            n7 = len(prev7)
            lines += [
                "",
                f"── PAST {n7} DAYS (up to {actual_date}) ──────────────────────",
                _temp_summary(prev7),
                _rain(prev7),
                _mean_pct(prev7, 'RH_avg'),
                f"  Frost days  : {_frost(prev7)}",
            ]

        if len(prev30) > len(prev7) + 3:  # only add 30-day if it meaningfully extends the 7-day window
            n30 = len(prev30)
            lines += [
                "",
                f"── PAST {n30} DAYS (up to {actual_date}) ──────────────────────",
                _temp_summary(prev30),
                _rain(prev30),
                _mean_pct(prev30, 'RH_avg'),
                f"  Frost days  : {_frost(prev30)}",
            ]

        lines += [
            "",
            "Use the above historical context for long-term agricultural planning decisions.",
        ]

        return "\n".join(l for l in lines if l is not None)
