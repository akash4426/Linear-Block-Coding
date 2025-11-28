import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# PAGE SETUP
# -------------------------
st.title("🔢 Linear Block Coding (Hamming 7,4) Simulator")
st.write("Interactive Streamlit App for Encoding, Decoding, Noise Simulation, and Visualization")

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def mod2(A): 
    return A % 2

def matmul_mod2(A, B): 
    return mod2(A @ B)

def encode(m, G):
    return matmul_mod2(m, G)

def syndrome(r, H):
    return matmul_mod2(r, H.T)

def build_syndrome_table(H):
    n = H.shape[1]
    table = {}
    for i in range(n):
        e = np.zeros((1,n), dtype=int)
        e[0,i] = 1
        s = tuple(syndrome(e, H).flatten())
        table[s] = e
    table[(0,0,0)] = np.zeros((1,n), dtype=int)
    return table

def decode(r, H, table):
    s = tuple(syndrome(r,H).flatten())
    e = table.get(s, np.zeros_like(r))  # fallback for unrecognized syndromes
    return mod2(r + e)

def flip_bits(c, p):
    """Random bit flipping based on noise probability p"""
    noise = (np.random.rand(*c.shape) < p).astype(int)
    return mod2(c + noise)

def flip_single_bit(c):
    """Flip exactly ONE random bit (always correctable)"""
    r = c.copy()
    bit = np.random.randint(0, 7)
    r[0, bit] ^= 1
    return r

# -------------------------
# HAMMING (7,4) MATRICES
# -------------------------
G = np.array([
    [1,0,0,0,0,1,1],
    [0,1,0,0,1,0,1],
    [0,0,1,0,1,1,0],
    [0,0,0,1,1,1,1]
], dtype=int)

H = np.array([
    [0,1,1,1,1,0,0],
    [1,0,1,1,0,1,0],
    [1,1,0,1,0,0,1]
], dtype=int)

syn_table = build_syndrome_table(H)

# -------------------------
# USER INPUT
# -------------------------
st.subheader("🧮 Enter 4-bit Message")
msg_bits = st.text_input("Example: 1011", "1011")

if len(msg_bits) != 4 or any(b not in "01" for b in msg_bits):
    st.error("Enter exactly 4 binary digits (0 or 1).")
    st.stop()

msg = np.array([[int(b) for b in msg_bits]])

st.write("### ✔️ Message Vector:", msg)

# Encode
codeword = encode(msg, G)
st.write("### 🟩 Encoded Codeword:", codeword)

# -------------------------
# ERROR MODE SELECTOR
# -------------------------
st.subheader("🎛️ Choose Error Mode")

error_mode = st.radio(
    "Select how the channel introduces errors:",
    ("Single-bit error (always correctable)", 
     "Random noise (may introduce multi-bit errors)")
)

p = 0
if error_mode == "Random noise (may introduce multi-bit errors)":
    p = st.slider("Bit Flip Probability (p):", 0.0, 0.5, 0.1)

# Apply Noise
if error_mode == "Single-bit error (always correctable)":
    received = flip_single_bit(codeword)
else:
    received = flip_bits(codeword, p)

st.write("### 📥 Received Vector:", received)

# -------------------------
# DECODING
# -------------------------
decoded = decode(received, H, syn_table)
st.write("### 🟦 Decoded Codeword:", decoded)

decoded_msg = decoded[0, :4]
st.write("### 🔍 Recovered Message:", decoded_msg)

# -------------------------
# RESULT MESSAGE
# -------------------------
if np.array_equal(msg, decoded_msg.reshape(1,4)):
    st.success("🎉 Message successfully recovered!")
else:
    st.warning("⚠️ Decoding failed. Hamming(7,4) corrects only 1-bit errors.\n"
               "This failure indicates multiple bits were flipped.")

# -------------------------
# PERFORMANCE SIMULATION
# -------------------------
st.subheader("📊 Performance Simulation & Visualization")

num_trials = st.slider("Number of trials:", 500, 5000, 1500)

probabilities = np.linspace(0, 0.5, 11)
success_rates = []
bit_error_rates = []

for prob in probabilities:
    success = 0
    bit_errors = 0
    total_bits = 0

    for _ in range(num_trials):
        m = np.random.randint(0,2,(1,4))
        c = encode(m, G)
        r = flip_bits(c, prob)
        d = decode(r, H, syn_table)

        total_bits += 4
        bit_errors += np.sum(d[0,:4] != m)
        success += int(np.array_equal(d[0,:4], m))

    success_rates.append(success/num_trials)
    bit_error_rates.append(bit_errors/total_bits)

# -------------------------
# PLOT 1: SUCCESS RATE
# -------------------------
fig1, ax1 = plt.subplots()
ax1.plot(probabilities, success_rates, marker='o')
ax1.set_title("Success Rate vs Bit Flip Probability")
ax1.set_xlabel("Noise Probability (p)")
ax1.set_ylabel("Success Rate")
ax1.grid(True)
st.pyplot(fig1)

# -------------------------
# PLOT 2: BER
# -------------------------
fig2, ax2 = plt.subplots()
ax2.plot(probabilities, bit_error_rates, marker='o')
ax2.set_title("BER vs Bit Flip Probability")
ax2.set_xlabel("Noise Probability (p)")
ax2.set_ylabel("Bit Error Rate")
ax2.grid(True)
st.pyplot(fig2)

st.success("Visualization completed! Try different error modes and noise settings above.")
