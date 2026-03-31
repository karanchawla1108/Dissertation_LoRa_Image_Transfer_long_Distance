
# Dissertation_LoRa_Image_Transfer_Long_Distance

## Enhancing Image Transmission over LoRa Networks Using Adaptive AI Compression — SF12 Long Range Module

**COM6016M Dissertation | York St John University | Karan Chawla | 2026**

This repository contains the SF12 long range transmission code for the VAE-based LoRa image transmission system. This module extends the base SF7 implementation ([available here](https://github.com/karanchawla1108/Dissertation_LoRa_Image_Transfer)) with improved spreading factor settings for increased range and better packet loss resilience.

---

## Overview

This module uses an **Improved VAE** with a 64-dimensional latent space (compared to 32 in the base model) and **SF12 spreading factor** with 250kHz bandwidth for extended range transmission. The system transmits compressed MNIST images over 433MHz LoRa using 6 packets instead of 3.

---

## LoRa Settings (SF12)

| Parameter | Value | Reason |
|-----------|-------|--------|
| Spreading Factor | SF12 | Maximum range |
| Bandwidth | 250 kHz | Better signal lock-on |
| Coding Rate | 8 | Higher error correction |
| Preamble Length | 32 | Reliable packet detection |
| TX Power | 23 dBm | Maximum power |
| Frequency | 433 MHz | Unlicensed band |
| CRC | Enabled | Error detection |


---

## Files

| File | Node | Description |
|------|------|-------------|
| `long_Distance_Sender.py` | Pi A (Sender) | Standard SF12 transmission |
| `long_distance_Receiver.py` | Pi B (Receiver) | Standard SF12 reception |
| `long_Distance_Sender_PacketLoss.py` | Pi A (Sender) | Packet loss tests (0%, 33%, 66%) |
| `Long_Distance_Receiver_PacketLoss.py` | Pi B (Receiver) | Packet loss test reception |

---

## Hardware Setup

| Component | Specification | Node |
|-----------|---------------|------|
| Raspberry Pi 4 Model B | 1.8GHz quad-core, 4GB RAM | Both |
| RFM9x LoRa Module | 433MHz, SX1276 chip | Both |
| INA219 Power Sensor | I2C, 12-bit ADC, address 0x40 | Both |

**Pi A (Sender):** username = karan, Python 3.11
**Pi B (Receiver):** username = ysj, Python 3.13

---

## Wiring

### RFM9x LoRa — Both Nodes
| RFM9x Pin | Pi Pin | GPIO | Function |
|-----------|--------|------|----------|
| VCC | Pin 1 | 3.3V | Power |
| GND | Pin 6 | GND | Ground |
| MOSI | Pin 19 | GPIO10 | SPI data out |
| MISO | Pin 21 | GPIO9 | SPI data in |
| SCK | Pin 23 | GPIO11 | SPI clock |
| CS | Pin 26 | CE1 (GPIO7) | Chip select |
| RST | Pin 22 | GPIO25 | Reset |
| DIO0 | Pin 18 | GPIO24 | Interrupt |




---
## Generating the Test Image

The MNIST digit 7 test image can be generated using the following command on Pi A:
```python
python3 -c "
import torchvision
import torchvision.transforms as transforms
from PIL import Image
dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True,
    transform=transforms.ToTensor()
)
img, label = dataset[0]
img_pil = Image.fromarray(
    (img.squeeze().numpy() * 255).astype('uint8')
)
img_pil.save('/home/karan/Desktop/Image_Disseratation/test_image.png')
print('Saved MNIST test image - digit: ' + str(label))
"
```

Saves to:
- Pi A: `/home/karan/Desktop/Image_Disseratation/test_image.png`
- Pi B: `/home/ysj/Image_dissertation/test_image.png`
## How to Run

### Standard Transmission

**Step 1 — Pi B (Receiver) first:**
```bash
python3 long_distance_Receiver.py
```
Wait for: `=== READY TO RECEIVE ===`

**Step 2 — Pi A (Sender):**
```bash
python3 long_Distance_Sender.py
```
Press ENTER when Pi B is ready.

---

### Packet Loss Tests

**Step 1 — Pi B (Receiver) first:**
```bash
python3 Long_Distance_Receiver_PacketLoss.py
```
Wait for: `Waiting for TEST 1...`

**Step 2 — Pi A (Sender):**
```bash
python3 long_Distance_Sender_PacketLoss.py
```
Press ENTER when Pi B is ready.

---

## Output Folders

**Standard test results:**
```
/home/ysj/Result_for_Image_Improved/
    run1/
        result.png
        result_big.png
        comparison.png
        results.txt
```

**Packet loss test results:**
```
/home/ysj/SF12_Packet_Loss_Results/
    run1/
        test1_0pct_loss_reconstructed.png
        test1_0pct_loss_comparison.png
        test2_33pct_loss_reconstructed.png
        test2_33pct_loss_comparison.png
        test3_66pct_loss_reconstructed.png
        test3_66pct_loss_comparison.png
        summary.txt
```

---

## Related Repositories

- **Base SF7 System:** [Dissertation_LoRa_Image_Transfer](https://github.com/karanchawla1108/Dissertation_LoRa_Image_Transfer)
- **VAE Model Training:** [Autoencoder-IoT-LoRa-Dissertation](https://github.com/karanchawla1108/Autoencoder-IoT-LoRa-Dissertation)

---

## Author

**Karan Chawla**
COM6016M Dissertation
York St John University
2026
