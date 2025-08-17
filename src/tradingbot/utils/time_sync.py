from __future__ import annotations

import ntplib


def check_ntp_offset(server: str = "pool.ntp.org") -> float:
    """Return clock offset in seconds with respect to an NTP server.

    Parameters
    ----------
    server:
        Hostname of the NTP server to query.
    Returns
    -------
    float
        Offset in seconds between the local clock and the server time.
    """
    client = ntplib.NTPClient()
    response = client.request(server, version=3)
    return response.offset
