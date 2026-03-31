# -*- coding: utf-8 -*-
# SF12 PACKET LOSS RECEIVER - Improved VAE (64 latent dim, 6 packets)
# Settings that WORK: BW=250kHz, CR=8, SF=12, preamble=32

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
from PIL import Image, ImageDraw
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
rfm.signal_bandwidth = 250000
rfm.coding_rate      = 8
rfm.spreading_factor = 12
rfm.enable_crc       = True
rfm.sync_word        = 0x12
rfm.preamble_length  = 32
print("LoRa ready - SF12 (BW=250kHz)")

# -----------------------------
# RUN FOLDER
# -----------------------------
BASE_DIR = '/home/ysj/SF12_Packet_Loss_Results'
os.makedirs(BASE_DIR, exist_ok=True)

run_num = 1
while os.path.exists(BASE_DIR + '/run' + str(run_num)):
    run_num += 1
run_folder = BASE_DIR + '/run' + str(run_num)
os.makedirs(run_folder)
print("Saving to: " + run_folder)

# -----------------------------
# FUNCTIONS
# -----------------------------
def receive_packets(num_packets=6, timeout=100):
    received = {}
    rssi_vals = []
    t_start = time.time()
    p_before = ina.power
    last_dot = time.time()

    while len(received) < num_packets:
        if time.time() - t_start > timeout:
            print("\nTimeout!")
            break

        pkt = rfm.receive(timeout=0.1)

        if pkt is not None and len(pkt) >= 3:
            idx = pkt[0]
            data = bytes(pkt[2:])
            rssi = rfm.last_rssi
            if idx not in received:
                received[idx] = data
                rssi_vals.append(rssi)
                print("\n[GOT] Packet " + str(idx+1) + "/" + str(num_packets) + " | RSSI: " + str(rssi) + " dBm | Total: " + str(len(received)) + "/" + str(num_packets))
            else:
                print("\n[DUP] Packet " + str(idx+1) + " ignored")

        if time.time() - last_dot > 1.0:
            print(".", end="", flush=True)
            last_dot = time.time()

        if len(received) == num_packets:
            print("\nAll packets received!")
            break

    rx_time = (time.time() - t_start) * 1000
    rx_power = (p_before + ina.power) / 2
    avg_rssi = round(sum(rssi_vals) / len(rssi_vals), 1) if rssi_vals else 0

    # Fill missing packets
    all_data = []
    lost = []
    for i in range(num_packets):
        if i in received:
            all_data.append(received[i])
        else:
            lost.append(i+1)
            size = 48 if i < num_packets - 1 else 16
            print("Packet " + str(i+1) + " LOST - zero fill")
            all_data.append(bytes(size))

    packets_received = len(received)
    print("Received: " + str(packets_received) + "/" + str(num_packets))
    print("Lost:     " + str(lost) if lost else "Lost:     none")
    print("RX time:  " + str(round(rx_time, 1)) + "ms")
    print("RX power: " + str(round(rx_power, 2)) + "mW")
    print("Avg RSSI: " + str(avg_rssi) + " dBm")

    return b''.join(all_data), packets_received, lost, rx_time, rx_power, avg_rssi

def decode_and_score(payload, test_num, loss_label, packets_received, lost):
    # Pad to 256 bytes
    if len(payload) < 256:
        payload += bytes(256 - len(payload))
    payload = payload[:256]

    latent = np.frombuffer(payload, dtype=np.float32).copy()
    z = torch.FloatTensor(latent).unsqueeze(0)

    t_start = time.time()
    p_before = ina.power
    with torch.no_grad():
        recon = model.decode(z)
    dec_time = (time.time() - t_start) * 1000
    dec_power = (p_before + ina.power) / 2

    img = recon.numpy().reshape(28, 28)
    print("Decoded in " + str(round(dec_time, 1)) + "ms | Power: " + str(round(dec_power, 2)) + "mW")

    # Save reconstructed
    output = (img * 255).astype(np.uint8)
    recon_path = run_folder + '/test' + str(test_num) + '_' + loss_label + '_reconstructed.png'
    Image.fromarray(output).save(recon_path)

    # SSIM
    try:
        orig = np.array(
            Image.open('/home/ysj/Image_dissertation/test_image.png').convert('L').resize((28, 28))
        ) / 255.0
        score = ssim(orig, img, data_range=1.0)
        print("SSIM: " + str(round(score, 4)))

        # Comparison image
        orig_big  = Image.fromarray((orig*255).astype(np.uint8)).resize((280,280), Image.NEAREST).convert('RGB')
        recon_big = Image.fromarray(output).resize((280,280), Image.NEAREST).convert('RGB')
        comp = Image.new('RGB', (600, 320), (40,40,40))
        comp.paste(orig_big,  (10, 20))
        comp.paste(recon_big, (310, 20))

        draw = ImageDraw.Draw(comp)
        title = "Test " + str(test_num) + " | " + loss_label + " | Pkts: " + str(packets_received) + "/6 | SSIM: " + str(round(score,4))
        draw.text((10, 5), title, fill=(255,255,100))
        if lost:
            draw.text((310, 5), "Lost: " + str(lost), fill=(255,100,100))

        comp_path = run_folder + '/test' + str(test_num) + '_' + loss_label + '_comparison.png'
        comp.save(comp_path)
        print("Comparison saved: " + comp_path)

    except Exception as e:
        score = 0.0
        print("SSIM error: " + str(e))

    return score, dec_time, dec_power

# -----------------------------
# MAIN â€” 3 TESTS
# -----------------------------
loss_labels = ['0pct_loss', '33pct_loss', '66pct_loss']
all_results = []

for test_num in range(1, 4):
    print("")
    print("==========================================")
    print("Waiting for TEST " + str(test_num) + " (" + loss_labels[test_num-1] + ") SF12...")
    print("==========================================")

    payload, pkts_rx, lost, rx_time, rx_power, avg_rssi = receive_packets()
    score, dec_time, dec_power = decode_and_score(
        payload, test_num, loss_labels[test_num-1], pkts_rx, lost
    )

    all_results.append({
        'test': test_num,
        'label': loss_labels[test_num-1],
        'received': pkts_rx,
        'lost': lost,
        'rx_time': rx_time,
        'rx_power': rx_power,
        'avg_rssi': avg_rssi,
        'dec_time': dec_time,
        'dec_power': dec_power,
        'ssim': score
    })

    print("Test " + str(test_num) + " complete.")
    if test_num < 3:
        print("Waiting for next test...")

# -----------------------------
# FINAL RESULTS
# -----------------------------
print("")
print("==========================================")
print("ALL TESTS COMPLETE - SF12 PACKET LOSS")
print("==========================================")
print("")
print("Test | Label       | Pkts RX | Lost   | Decode  | RSSI    | SSIM")
print("-----|-------------|---------|--------|---------|---------|------")
for r in all_results:
    lost_str = str(r['lost']) if r['lost'] else "none"
    print(str(r['test']) + "    | " + r['label'] + " | " + str(r['received']) + "/6     | " + lost_str + " | " + str(round(r['dec_time'],1)) + "ms   | " + str(r['avg_rssi']) + " dBm | " + str(round(r['ssim'],4)))

print("")
print("Results saved to: " + run_folder)

# Save summary log
with open(run_folder + '/summary.txt', 'w') as f:
    f.write("SF12 Packet Loss Test Results\n")
    f.write("==============================\n")
    f.write("Spreading Factor: SF12\n")
    f.write("Bandwidth:        250 kHz\n")
    f.write("Coding Rate:      8\n")
    f.write("Preamble:         32\n\n")
    for r in all_results:
        f.write("Test " + str(r['test']) + " | " + r['label'] + "\n")
        f.write("  Packets RX: " + str(r['received']) + "/6\n")
        f.write("  Lost:       " + str(r['lost']) + "\n")
        f.write("  RX time:    " + str(round(r['rx_time'],1)) + "ms\n")
        f.write("  Decode:     " + str(round(r['dec_time'],1)) + "ms\n")
        f.write("  RX power:   " + str(round(r['rx_power'],2)) + "mW\n")
        f.write("  RSSI:       " + str(r['avg_rssi']) + " dBm\n")
        f.write("  SSIM:       " + str(round(r['ssim'],4)) + "\n\n")

print("Summary log saved!")
print("Done!")
