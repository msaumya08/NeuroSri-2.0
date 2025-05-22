import asyncio
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from bleak import BleakScanner, BleakClient
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the prediction module
try:
    from realtime_prediction import process_new_data
    PREDICTION_ENABLED = True
except ImportError:
    print("Warning: realtime_prediction.py not found. Running without predictions.")
    PREDICTION_ENABLED = False

# BLE and data parameters
DEVICE_NAME_PREFIX = "NPG"
DATA_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
CONTROL_CHAR_UUID = "0000ff01-0000-1000-8000-00805f9b34fb"
SINGLE_SAMPLE_LEN = 7
BLOCK_COUNT = 10
NEW_PACKET_LEN = SINGLE_SAMPLE_LEN * BLOCK_COUNT
SAMPLE_RATE = 250  # Hz (adjust if needed)
WINDOW_SEC = 10
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SEC

# Buffers for 3 channels
eeg_buffers = [[], [], []]
running = True  # <-- Add this flag

async def main():
    global running
    print("Scanning for NPG BLE devices...")
    devices = await BleakScanner.discover(timeout=10.0)
    npg_devices = [d for d in devices if d.name and d.name.startswith(DEVICE_NAME_PREFIX)]
    if not npg_devices:
        print("No NPG BLE devices found!")
        return
    device = npg_devices[0]
    print(f"Connecting to {device.name} ({device.address})...")

    async with BleakClient(device) as client:
        await client.write_gatt_char(CONTROL_CHAR_UUID, b"START", response=True)
        print("Connected and sent START command.")

        with open("eeg_data.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "ch1", "ch2", "ch3"])

            plt.style.use('dark_background')
            plt.ion()
            fig, axs = plt.subplots(3, 1, figsize=(12, 8))
            fig.suptitle("Real-Time EEG Waves for 3 Channels", fontsize=18, fontweight='bold')
            channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

            def update_plots():
                for ch in range(3):
                    buf = eeg_buffers[ch]
                    ax_eeg = axs[ch]
                    ax_eeg.clear()
                    if len(buf) > 100:  # Only plot if enough data
                        t = np.linspace(-len(buf)/SAMPLE_RATE, 0, len(buf))
                        ax_eeg.plot(t, buf, color=channel_colors[ch], linewidth=1.5)
                        ax_eeg.set_title(f"EEG Ch{ch+1}", fontsize=14, fontweight='bold')
                        ax_eeg.set_ylabel("Value", fontsize=12)
                        ax_eeg.set_xlabel("Time (s)", fontsize=12)
                        ax_eeg.grid(True, linestyle='--', alpha=0.5)
                        ax_eeg.set_xlim([t[0], 0])
                        # Dynamic y-limits
                        min_y = min(buf)
                        max_y = max(buf)
                        margin = (max_y - min_y) * 0.1 if max_y != min_y else 1
                        ax_eeg.set_ylim([min_y - margin, max_y + margin])
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.pause(0.001)
                print("Plot updated, buffer lengths:", [len(b) for b in eeg_buffers])  # Debug print

            def handle_notification(sender, data):
                if not running:
                    return
                now = time.time()
                if len(data) == NEW_PACKET_LEN:
                    for i in range(0, NEW_PACKET_LEN, SINGLE_SAMPLE_LEN):
                        sample = data[i:i+SINGLE_SAMPLE_LEN]
                        if len(sample) == SINGLE_SAMPLE_LEN:
                            ch1 = int.from_bytes(sample[1:3], byteorder='big', signed=True)
                            ch2 = int.from_bytes(sample[3:5], byteorder='big', signed=True)
                            ch3 = int.from_bytes(sample[5:7], byteorder='big', signed=True)
                            
                            # Send data to prediction module if enabled
                            if PREDICTION_ENABLED:
                                process_new_data(ch1, ch2, ch3)
                            
                            print(f"{now:.2f}, {ch1}, {ch2}, {ch3}")
                            writer.writerow([now, ch1, ch2, ch3])
                            for idx, val in enumerate([ch1, ch2, ch3]):
                                eeg_buffers[idx].append(val)
                                if len(eeg_buffers[idx]) > WINDOW_SIZE:
                                    eeg_buffers[idx] = eeg_buffers[idx][-WINDOW_SIZE:]
                elif len(data) == SINGLE_SAMPLE_LEN:
                    ch1 = int.from_bytes(data[1:3], byteorder='big', signed=True)
                    ch2 = int.from_bytes(data[3:5], byteorder='big', signed=True)
                    ch3 = int.from_bytes(data[5:7], byteorder='big', signed=True)
                    
                    # Send data to prediction module if enabled
                    if PREDICTION_ENABLED:
                        process_new_data(ch1, ch2, ch3)
                    
                    print(f"{now:.2f}, {ch1}, {ch2}, {ch3}")
                    writer.writerow([now, ch1, ch2, ch3])
                    for idx, val in enumerate([ch1, ch2, ch3]):
                        eeg_buffers[idx].append(val)
                        if len(eeg_buffers[idx]) > WINDOW_SIZE:
                            eeg_buffers[idx] = eeg_buffers[idx][-WINDOW_SIZE:]
                f.flush()
                update_plots()

            await client.start_notify(DATA_CHAR_UUID, handle_notification)
            print("Receiving data... Close the plot window or press Ctrl+C to stop.")
            try:
                while plt.fignum_exists(fig.number):
                    await asyncio.sleep(0.1)
            except KeyboardInterrupt:
                print("Stopped.")
            finally:
                running = False
                await client.stop_notify(DATA_CHAR_UUID)
                await asyncio.sleep(0.2)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nProgram terminated by user.")

        
