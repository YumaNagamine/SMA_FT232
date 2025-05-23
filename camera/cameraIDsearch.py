# -*- coding: utf-8 -*-

# CODE BY: Renke. Askar. LIU, by the help of ChatGPT
# DATE: 20250518
# VERSION: 1.0
 
import wmi

c = wmi.WMI()
for dev in c.Win32_PnPEntity():
    name = getattr(dev, 'Name', None)
    pnp  = getattr(dev, 'PNPDeviceID', None)
    # Skip entries without a name or ID
    if not name or not pnp:
        continue
    # Filter to typical camera keywords
    lower = name.lower()
    if 'camera' in lower or 'imaging' in lower in lower:
        print(f"Name: {name}\nPNPDeviceID: {pnp}\n")


# PNPDeviceID: USB\VID_32E4&PID_0234&MI_00\7&1083005A&0&0000
# Name: Global Shutter Camera
# PNPDeviceID: USB\VID_32E4&PID_0234&MI_00\6&325DB0BA&0&0000