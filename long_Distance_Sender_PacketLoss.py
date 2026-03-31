# -*- coding: utf-8 -*-
# SF12 PACKET LOSS SENDER - Improved VAE (64 latent dim, 6 packets)
# Settings that WORK: BW=250kHz, CR=8, SF=12, preamble=32

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
# LORA SETUP
# -----------------------------
spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
cs  = digitalio.DigitalInOut(board.CE1)
cs.direction = digitalio.Direction.OUTPUT
rst = digitalio.DigitalInOut(board.D25)
rst.direction = digitalio.Direction.OUTPUT

rst.value = False
time.sleep(0.1)
rst.value = True
time.sleep(0.1)

rfm = adafruit_rfm9x.RFM9x(spi, cs, rst, 433.0)
rfm.tx_power         = 23
rfm.signal_bandwidth = 250000
rfm.coding_rate      = 8
rfm.spreading_factor = 12
rfm.enable_crc       = True
rfm.sync_word        = 0x12
rfm.preamble_length  = 32
print("LoRa ready - SF12 (BW=250kHz)")

# -----------------------------
# FUNCTIONS
# -----------------------------
def prepare_image(path):
    img = Image.open(path).convert('L').resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.FloatTensor(arr).unsqueeze(0).unsqueeze(0)

def encode_image(tensor):
    t_start = time.time()
    p_before = ina.power
    with torch.no_grad():
        mean, var = model.encode(tensor)
        z = model.reparameterise(mean, var)
    enc_time = (time.time() - t_start) * 1000
    enc_power = (p_before + ina.power) / 2
    print("Encoded in " + str(round(enc_time, 1)) + "ms | Power: " + str(round(enc_power, 2)) + "mW")
    return z.numpy().flatten(), enc_time, enc_power

def split_packets(latent):
    payload = latent.astype(np.float32).tobytes()
    packets = [payload[i:i+48] for i in range(0, len(payload), 48)]
    print("Payload: " + str(len(payload)) + " bytes â†’ " + str(len(packets)) + " packets")
    return packets

def send_packets(packets, drop_list=[]):
    print("Sending " + str(len(packets)) + " packets - dropping: " + str(drop_list))
    t_start = time.time()
    p_before = ina.power
    sent = 0
    dropped = 0

    time.sleep(2.0)  # Short pause before first packet

    for i, pkt in enumerate(packets):
        if (i+1) in drop_list:
            print("  Packet " + str(i+1) + "/" + str(len(packets)) + " DROPPED (simulated loss)")
            dropped += 1
        else:
            header = bytes([i, len(packets)])
            rfm.send(header + pkt)
            print("  Sent packet " + str(i+1) + "/" + str(len(packets)))
            sent += 1
        if i < len(packets) - 1:
            time.sleep(10.0)

    tx_time = (time.time() - t_start) * 1000
    tx_power = (p_before + ina.power) / 2
    print("Sent: " + str(sent) + "/" + str(len(packets)) + " | Dropped: " + str(dropped) + "/" + str(len(packets)))
    print("TX time: " + str(round(tx_time, 1)) + "ms | Power: " + str(round(tx_power, 2)) + "mW")
    return tx_time, tx_power, sent, dropped

def run_test(test_num, drop_list, label, image_path):
    print("")
    print("==========================================")
    print("TEST " + str(test_num) + " - " + label)
    print("==========================================")
    tensor = prepare_image(image_path)
    latent, enc_time, enc_power = encode_image(tensor)
    packets = split_packets(latent)
    tx_time, tx_power, sent, dropped = send_packets(packets, drop_list=drop_list)
    print("Test " + str(test_num) + " done. Waiting 60 seconds for receiver...")
    time.sleep(30)
    return enc_time, enc_power, tx_time, tx_power, sent, dropped

# -----------------------------
# MAIN
# -----------------------------
image_path = '/home/karan/Desktop/Image_Disseratation/test_image.png'

print("")
print("==========================================")
print("Run receiver on Pi B FIRST!")
print("Wait until Pi B shows:")
print("  'Waiting for TEST 1...'")
print("==========================================")
input("Then press ENTER here to start...")

results = []
tests = [
    (1, [],     "0%  loss - send all 6 packets"),
    (2, [2],    "33% loss - drop packet 2"),
    (3, [1, 2], "66% loss - drop packets 1 and 2"),
]

for test_num, drop_list, label in tests:
    enc_time, enc_power, tx_time, tx_power, sent, dropped = run_test(
        test_num, drop_list, label, image_path
    )
    results.append({
        'test': test_num,
        'label': label,
        'sent': sent,
        'dropped': dropped,
        'enc_time': enc_time,
        'enc_power': enc_power,
        'tx_time': tx_time,
        'tx_power': tx_power
    })

print("")
print("==========================================")
print("ALL TESTS COMPLETE - SUMMARY (SF12)")
print("==========================================")
for r in results:
    print("Test " + str(r['test']) + " | " + r['label'] + " | Sent: " + str(r['sent']) + "/6 | TX: " + str(round(r['tx_time'], 1)) + "ms")
print("Check Pi B for SSIM scores!")
