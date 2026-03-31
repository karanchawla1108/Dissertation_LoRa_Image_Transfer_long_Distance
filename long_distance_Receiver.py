# -*- coding: utf-8 -*-
# SF12 LONG DISTANCE RECEIVER - Improved VAE (64 latent dim, 6 packets)
# Settings that WORK: BW=250kHz, CR=8, SF=12, preamble=32, fast poll

import torch
import torch.nn as nn
import numpy as np
import time
import os
import busio
import board
import adafruit_ina219
import adafruit_rfm9x
import digitalio
from PIL import Image
from skimage.metrics import structural_similarity as ssim

LATENT_DIM = 64

class ImprovedVAE(nn.Module):
    def __init__(self):
        super(ImprovedVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.mean_layer = nn.Linear(128, LATENT_DIM)
        self.var_layer  = nn.Linear(128, LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid()
        )
    def encode(self, x):
        x = self.encoder(x)
        return self.mean_layer(x), self.var_layer(x)
    def reparameterise(self, mean, var):
        return mean + var * torch.randn_like(var)
    def decode(self, z):
        return self.decoder(z)
    def forward(self, x):
        mean, var = self.encode(x)
        z = self.reparameterise(mean, var)
        return self.decode(z), mean, var

# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading improved VAE model...")
model = ImprovedVAE()
model.load_state_dict(torch.load(
    '/home/ysj/Desktop/New Improved MNIST VAE model/vae_model_improved.pth',
    map_location='cpu'
))
model.eval()
print("Model loaded OK")

# -----------------------------
# INA219
# -----------------------------
i2c = busio.I2C(board.SCL, board.SDA)
ina = adafruit_ina219.INA219(i2c)
print("INA219 ready")
print("Idle voltage: " + str(round(ina.bus_voltage, 2)) + "V")
print("Idle current: " + str(round(ina.current, 2)) + "mA")
print("Idle power:   " + str(round(ina.power, 2)) + "mW")

# -----------------------------
# LORA SETUP â€” Settings that WORK
# -----------------------------
spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
cs  = digitalio.DigitalInOut(board.CE1)
cs.direction = digitalio.Direction.OUTPUT
rst = digitalio.DigitalInOut(board.D25)
rst.direction = digitalio.Direction.OUTPUT

# Hard reset
rst.value = False
time.sleep(0.1)
rst.value = True
time.sleep(0.1)

rfm = adafruit_rfm9x.RFM9x(spi, cs, rst, 433.0)
rfm.signal_bandwidth = 250000  # Match sender
rfm.coding_rate      = 8       # Match sender
rfm.spreading_factor = 12      # Match sender
rfm.enable_crc       = True
rfm.sync_word        = 0x12    # Match sender
rfm.preamble_length  = 32      # Match sender
print("LoRa ready - SF12 long range (BW=250kHz)")

# -----------------------------
# RUN FOLDER
# -----------------------------
BASE_DIR = '/home/ysj/Result_for_Image_Improved_LongDistance'
os.makedirs(BASE_DIR, exist_ok=True)

run_num = 1
while os.path.exists(BASE_DIR + '/run' + str(run_num)):
    run_num += 1
run_folder = BASE_DIR + '/run' + str(run_num)
os.makedirs(run_folder)
print("Saving to: " + run_folder)

# -----------------------------
# RECEIVE PACKETS
# -----------------------------
NUM_PACKETS = 6
TIMEOUT     = 600  # 10 minutes timeout

received = {}
rssi_values = []

print("")
print("=== READY TO RECEIVE ===")
print("Now press ENTER on Pi A sender")
print("")

t_start = time.time()
p_rx_before = ina.power
last_dot = time.time()

while len(received) < NUM_PACKETS:
    if time.time() - t_start > TIMEOUT:
        print("\nOverall timeout reached")
        break

    # Fast poll â€” 0.1s timeout catches all packets
    packet = rfm.receive(timeout=0.1)

    if packet is not None and len(packet) >= 3:
        pkt_id = packet[0]
        data   = bytes(packet[2:])
        rssi   = rfm.last_rssi

        if pkt_id not in received:
            received[pkt_id] = data
            rssi_values.append(rssi)
            print("\n[RECEIVED] Packet " + str(pkt_id+1) + "/6 | Total: " + str(len(received)) + "/6 | RSSI: " + str(rssi) + " dBm")
        else:
            print("\n[DUPLICATE] Packet " + str(pkt_id+1) + " ignored")

    # Heartbeat dot every second
    if time.time() - last_dot > 1.0:
        print(".", end="", flush=True)
        last_dot = time.time()

    if len(received) == NUM_PACKETS:
        print("\nAll 6 packets received!")
        break

p_rx_after = ina.power
rx_time = (time.time() - t_start) * 1000
rx_power = (p_rx_before + p_rx_after) / 2
avg_rssi = round(sum(rssi_values) / len(rssi_values), 1) if rssi_values else 0

print("")
print("Received: " + str(len(received)) + "/" + str(NUM_PACKETS) + " packets")
print("RX time:  " + str(round(rx_time, 1)) + "ms")
print("RX power: " + str(round(rx_power, 2)) + "mW")
if rssi_values:
    print("Avg RSSI: " + str(avg_rssi) + " dBm")

# -----------------------------
# REASSEMBLE PAYLOAD
# -----------------------------
all_data = []
for i in range(NUM_PACKETS):
    if i in received:
        all_data.append(received[i])
    else:
        size = 48 if i < NUM_PACKETS - 1 else 16
        print("Packet " + str(i+1) + " LOST - zero fill (" + str(size) + " bytes)")
        all_data.append(bytes(size))

payload = b''.join(all_data)
if len(payload) < 256:
    payload += bytes(256 - len(payload))
payload = payload[:256]

# -----------------------------
# DECODE IMAGE
# -----------------------------
print("Decoding...")
latent = np.frombuffer(payload, dtype=np.float32).copy()
z = torch.FloatTensor(latent).unsqueeze(0)

t_dec_start = time.time()
p_dec_before = ina.power
with torch.no_grad():
    recon = model.decode(z)
p_dec_after = ina.power
dec_time = (time.time() - t_dec_start) * 1000
dec_power = (p_dec_before + p_dec_after) / 2

img_array = recon.numpy().reshape(28, 28)
print("Decoded in " + str(round(dec_time, 1)) + "ms | Power: " + str(round(dec_power, 2)) + "mW")

# -----------------------------
# SAVE IMAGES
# -----------------------------
output = (img_array * 255).astype(np.uint8)
Image.fromarray(output).save(run_folder + '/result.png')
Image.fromarray(output).resize((280, 280), Image.NEAREST).save(run_folder + '/result_big.png')
print("Saved result images")

# -----------------------------
# SSIM SCORE
# -----------------------------
try:
    orig = np.array(
        Image.open('/home/ysj/Image_dissertation/test_image.png').convert('L').resize((28, 28))
    ) / 255.0
    score = ssim(orig, img_array, data_range=1.0)
    print("SSIM Score: " + str(round(score, 4)))

    # Comparison image
    orig_big  = Image.fromarray((orig * 255).astype(np.uint8)).resize((280, 280), Image.NEAREST)
    recon_big = Image.fromarray(output).resize((280, 280), Image.NEAREST)
    comp = Image.new('RGB', (600, 320), (40, 40, 40))
    comp.paste(orig_big.convert('RGB'),  (10, 20))
    comp.paste(recon_big.convert('RGB'), (310, 20))
    comp.save(run_folder + '/comparison.png')
    print("Comparison saved")

except Exception as e:
    score = 0.0
    print("SSIM error: " + str(e))

# -----------------------------
# SAVE LOG
# -----------------------------
total_time = rx_time + dec_time
with open(run_folder + '/results.txt', 'w') as f:
    f.write("VAE LoRa Long Distance Results (SF12)\n")
    f.write("======================================\n")
    f.write("Spreading Factor:  SF12\n")
    f.write("Bandwidth:         250 kHz\n")
    f.write("Coding Rate:       8\n")
    f.write("Preamble:          32\n")
    f.write("Packets received:  " + str(len(received)) + "/6\n")
    f.write("RX time:           " + str(round(rx_time, 1)) + "ms\n")
    f.write("Decode time:       " + str(round(dec_time, 1)) + "ms\n")
    f.write("Total latency:     " + str(round(total_time, 1)) + "ms\n")
    f.write("RX power:          " + str(round(rx_power, 2)) + "mW\n")
    f.write("Decode power:      " + str(round(dec_power, 2)) + "mW\n")
    f.write("Avg RSSI:          " + str(avg_rssi) + " dBm\n")
    f.write("SSIM Score:        " + str(round(score, 4)) + "\n")
print("Log saved")

# -----------------------------
# FINAL RESULTS
# -----------------------------
print("")
print("==========================================")
print("FINAL RESULTS")
print("==========================================")
print("Run folder:       " + run_folder)
print("Packets received: " + str(len(received)) + "/6")
print("RX time:          " + str(round(rx_time, 1)) + "ms")
print("Decode time:      " + str(round(dec_time, 1)) + "ms")
print("Total latency:    " + str(round(total_time, 1)) + "ms")
print("RX power:         " + str(round(rx_power, 2)) + "mW")
print("Decode power:     " + str(round(dec_power, 2)) + "mW")
print("Avg RSSI:         " + str(avg_rssi) + " dBm")
print("SSIM Score:       " + str(round(score, 4)))
print("Done!")
