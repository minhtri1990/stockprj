def format_number(val):
    try:
        if val is None:
            return "-"
        if abs(val) >= 1_000_000_000:
            return f"{val/1_000_000_000:.2f}B"
        if abs(val) >= 1_000_000:
            return f"{val/1_000_000:.2f}M"
        if abs(val) >= 1_000:
            return f"{val/1_000:.2f}K"
        return f"{val:,.2f}"
    except Exception:
        return str(val)
