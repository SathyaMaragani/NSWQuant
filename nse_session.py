"""Local NSE index futures session schedule in IST."""

import datetime as dt

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

# NSE F&O circulars 71777 and 72262.
HOLIDAYS = frozenset(map(dt.date.fromisoformat, (
    "2026-01-15", "2026-01-26", "2026-03-03", "2026-03-26", "2026-03-31",
    "2026-04-03", "2026-04-14", "2026-05-01", "2026-05-28", "2026-06-26",
    "2026-09-14", "2026-10-02", "2026-10-20", "2026-11-10", "2026-11-24", "2026-12-25",
)))

# Budget and Muhurat sessions, NSE F&O circulars 72352 and 76940.
SPECIAL_SESSIONS = {
    dt.date(2026, 2, 1): (dt.time(9, 15), dt.time(15, 30)),
    dt.date(2026, 11, 8): (dt.time(14, 20), dt.time(15, 40)),
}


def session_hours(day, expiry=None):
    """Return the continuous trading window for the given date and contract."""
    if day in SPECIAL_SESSIONS:
        return SPECIAL_SESSIONS[day]
    if day.weekday() >= 5 or day in HOLIDAYS:
        return None
    # NSE F&O circulars 74467 and 77356.
    if day >= dt.date(2026, 12, 7):
        close = dt.time(15, 30) if day.isoformat() == expiry else dt.time(15, 45)
    elif day >= dt.date(2026, 8, 3):
        close = dt.time(15, 40)
    else:
        close = dt.time(15, 30)
    return dt.time(9, 15), close


def next_open(now):
    day = now.date()
    while True:
        hours = session_hours(day)
        if hours:
            opening = dt.datetime.combine(day, hours[0], IST)
            if opening > now:
                return opening
        day += dt.timedelta(days=1)
