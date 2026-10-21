"""APP-350 data-type labels -> Google Play Data Safety categories.

APP-350 only labels identifiers, contact/demographic data, address books and location,
so only 4 of Play's 14 categories can be scored against it. Labels with no clean Play
equivalent map to None and are left out of the gold set (not counted as right or wrong).
"""

APP350_TO_PLAY: dict[str, str | None] = {
    # Personal info: name, email, phone, address, age, gender...
    "Contact": "Personal info",
    "Contact_City": "Personal info",
    "Contact_E_Mail_Address": "Personal info",
    "Contact_Phone_Number": "Personal info",
    "Contact_Postal_Address": "Personal info",
    "Contact_ZIP": "Personal info",
    "Demographic": "Personal info",
    "Demographic_Age": "Personal info",
    "Demographic_Gender": "Personal info",
    # Play has no slot for passwords; they're account credentials, not declared data.
    "Contact_Password": None,
    # The user's phone contacts list.
    "Contact_Address_Book": "Contacts",
    # Device or other IDs: anything that identifies a device, browser or app install.
    "Identifier": "Device or other IDs",
    "Identifier_Ad_ID": "Device or other IDs",
    "Identifier_Cookie_or_similar_Tech": "Device or other IDs",
    "Identifier_Device_ID": "Device or other IDs",
    "Identifier_IMEI": "Device or other IDs",
    "Identifier_IMSI": "Device or other IDs",
    "Identifier_IP_Address": "Device or other IDs",
    "Identifier_MAC": "Device or other IDs",
    "Identifier_SIM_Serial": "Device or other IDs",
    # Ambiguous under Play's definitions (carrier is network info, SSID can imply location).
    "Identifier_Mobile_Carrier": None,
    "Identifier_SSID_BSSID": None,
    # Location, any precision.
    "Location": "Location",
    "Location_Bluetooth": "Location",
    "Location_Cell_Tower": "Location",
    "Location_GPS": "Location",
    "Location_IP_Address": "Location",
    "Location_WiFi": "Location",
    # Sign-in providers: a data flow, not a data type.
    "SSO": None,
    "Facebook_SSO": None,
}

# The Play categories APP-350 can actually test. Precision/recall outside these is meaningless.
EVALUABLE = sorted({v for v in APP350_TO_PLAY.values() if v})


def split_practice(practice: str) -> tuple[str, str]:
    """'Location_GPS_3rdParty' -> ('Location_GPS', '3rdParty'). SSO labels have no party suffix."""
    for party in ("1stParty", "3rdParty"):
        if practice.endswith("_" + party):
            return practice[: -len(party) - 1], party
    return practice, "3rdParty"
