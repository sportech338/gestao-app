def parse_hour_bucket(h):
    if h is None: return None
    try:
        s=str(h).strip()
        val=int(s.split(":")[0]) if ":" in s else int(float(s))
        return max(0, min(23, val))
    except: return None
