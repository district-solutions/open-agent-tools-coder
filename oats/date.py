#!/usr/bin/env python3

from datetime import datetime, timezone, timedelta

def utc() -> datetime:
    """
    get the utc datetime
    """
    return datetime.now(timezone.utc)

def naive() -> datetime:
    """
    get naive datetime from utc

    # Timezone Aware UTC

    print(datetime.now(timezone.utc))
    2025-10-05 15:11:31.981400+00:00

    # Return as Timezone Naive Datetime

    print(datetime.now(timezone.utc).replace(tzinfo=None))
    2025-10-05 15:13:54.423262
    """
    return utc().replace(tzinfo=None)

def get_third_friday_dates():
    """
    Generate a list of the third Friday dates for the next 6 months.
    Returns dates in YYYYMMDD format, space-delimited.
    """
    # Get current date
    today = naive().today()

    # Calculate the date 6 months from now
    """
    if today.month + 6 <= 12:
        six_months_later = today.replace(month=today.month + 6)
    else:
        six_months_later = today.replace(year=today.year + 1, month=today.month + 6 - 12)
    """

    third_fridays = []

    # Start from the first day of the current month
    current_month = today.replace(day=1)

    # For each month in the next 6 months
    for i in range(6):
        # Calculate target month
        if current_month.month + i <= 12:
            target_month = current_month.replace(month=current_month.month + i)
        else:
            target_month = current_month.replace(year=current_month.year + 1, month=current_month.month + i - 12)

        # Find first day of the month
        first_day = target_month.replace(day=1)

        # Find the first Friday of the month
        # weekday(): 0=Monday, 1=Tuesday, ..., 6=Sunday
        # We want Friday (weekday 4)
        days_to_add = (4 - first_day.weekday()) % 7  # Days to add to get to first Friday
        first_friday = first_day + timedelta(days=days_to_add)

        # Third Friday is 14 days after first Friday
        third_friday = first_friday + timedelta(days=14)

        # Ensure it's actually in the target month
        if third_friday.month == target_month.month:
            third_fridays.append(third_friday)

    # Format as YYYYMMDD and return space-delimited string
    formatted_dates = [date.strftime("%Y%m%d") for date in third_fridays]
    return " ".join(formatted_dates)

def get_utc_str() -> str:
    """
    get the utc datetime
    """
    return utc().strftime('%Y-%m-%d %H:%M:%S')

if __name__ == "__main__":
    print(get_third_friday_dates())
