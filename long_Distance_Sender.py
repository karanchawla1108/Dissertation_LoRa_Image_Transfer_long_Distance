# -*- coding: utf-8 -*-
# SF12 LONG DISTANCE SENDER - Improved VAE (64 latent dim, 6 packets)
# Settings that WORK: BW=250kHz, CR=8, SF=12, preamble=32, 10s delay

import torch
import torch.nn as nn
import numpy as np
import time
import busio
import board
import adafruit_ina219
import adafruit_rfm9x
import digitalio
from PIL import Image

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
    '/home/karan/Desktop/New Improved MNIST VAE model/vae_model_improved.pth',
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
rfm.tx_power         = 23
rfm.signal_bandwidth = 250000  # Wider BW for reliable lock-on
rfm.coding_rate      = 8       # Higher coding rate for robustness
rfm.spreading_factor = 12      # SF12 for long range
rfm.enable_crc       = True
rfm.sync_word        = 0x12
rfm.preamble_length  = 32      # Longer preamble for reliable detection
print("LoRa ready - SF12 long range (BW=250kHz)")

# -----------------------------
# WAIT FOR USER
# -----------------------------
print("")
print("==========================================")
print("Run receiver on Pi B FIRST!")
print("Wait until Pi B shows:")
print("  '=== READY TO RECEIVE ==='")
print("==========================================")
input("Then press ENTER here to start...")
print("")

# -----------------------------
# ENCODE IMAGE
# -----------------------------
print("Encoding image...")
t_enc_start = time.time()
p_enc_before = ina.power

img = Image.open('/home/karan/Desktop/Image_Disseratation/test_image.png').convert('L').resize((28, 28))
arr = np.array(img, dtype=np.float32) / 255.0
tensor = torch.FloatTensor(arr).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    mean, var = model.encode(tensor)
    z = model.reparameterise(mean, var)
    latent = z.numpy().flatten()

p_enc_after = ina.power
enc_time = (time.time() - t_enc_start) * 1000
enc_power = (p_enc_before + p_enc_after) / 2
print("Encoded in " + str(round(enc_time, 1)) + "ms | Power: " + str(round(enc_power, 2)) + "mW")

# -----------------------------
# SPLIT INTO PACKETS
# -----------------------------
payload = latent.astype(np.float32).tobytes()
packets = [payload[i:i+48] for i in range(0, len(payload), 48)]
print("Payload: " + str(len(payload)) + " bytes â†’ " + str(len(packets)) + " packets")

# -----------------------------
# SEND PACKETS
# -----------------------------
print("")
print("--- TRANSMITTING ---")
t_tx_start = time.time()
p_tx_before = ina.power

time.sleep(2.0)  # Short pause before first packet

for i, packet in enumerate(packets):
    header = bytes([i, len(packets)])
    rfm.send(header + packet)
    print("Sent packet " + str(i+1) + "/" + str(len(packets)))
    if i < len(packets) - 1:
        time.sleep(10.0)  # 10 second gap between packets

p_tx_after = ina.power
tx_time = (time.time() - t_tx_start) * 1000
tx_power = (p_tx_before + p_tx_after) / 2

# -----------------------------
# RESULTS
# -----------------------------
print("")
print("==========================================")
print("ALL PACKETS SENT - RESULTS SUMMARY")
print("==========================================")
print("Spreading Factor: SF12")
print("Bandwidth:        250 kHz")
print("Coding Rate:      8")
print("Preamble:         32")
print("Packets sent:     " + str(len(packets)) + "/6")
print("Payload size:     " + str(len(payload)) + " bytes")
print("Encode time:      " + str(round(enc_time, 1)) + "ms")
print("TX time:          " + str(round(tx_time, 1)) + "ms")
print("Total time:       " + str(round(enc_time + tx_time, 1)) + "ms")
print("Encode power:     " + str(round(enc_power, 2)) + "mW")
print("TX power:         " + str(round(tx_power, 2)) + "mW")
print("Check Pi B for SSIM score!")
print("Done!")
