import streamlit as st
import numpy as np
import pandas as pd
import heapq
import pickle
import io
import time
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px


# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Image Compression using Huffman Coding",
    page_icon="🖼️",
    layout="wide"
)


# -------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eaf3ff 0%, #ffffff 45%, #f5ecff 100%);
}
.main-title {
    text-align: center;
    font-size: 46px;
    font-weight: 900;
    color: #102a63;
    font-family: Georgia, serif;
}
.sub-title {
    text-align: center;
    font-size: 19px;
    color: #444;
}
.card {
    padding: 20px;
    background: white;
    border-radius: 18px;
    box-shadow: 0px 4px 16px rgba(0,0,0,0.10);
    border-left: 7px solid #1f77ff;
    margin-bottom: 12px;
}
.step-card {
    padding: 16px;
    background: #f8fbff;
    border-radius: 14px;
    border-left: 6px solid #0d6efd;
    margin-bottom: 10px;
}
.green-card {
    padding: 16px;
    background: #eaf8ef;
    border-radius: 14px;
    border-left: 6px solid #198754;
}
.red-card {
    padding: 16px;
    background: #fff0f0;
    border-radius: 14px;
    border-left: 6px solid #dc3545;
}
.info-card {
    padding: 16px;
    background: #fff8e6;
    border-radius: 14px;
    border-left: 6px solid #ffb703;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# TITLE
# -------------------------------------------------------
st.markdown("<div class='main-title'>Image Compression using Huffman Coding</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Advanced Color Image Compression Web App using RGB Channel-wise Huffman Coding</div>",
    unsafe_allow_html=True
)
st.write("---")


# -------------------------------------------------------
# HUFFMAN NODE
# -------------------------------------------------------
class HuffmanNode:
    def __init__(self, pixel=None, frequency=0):
        self.pixel = pixel
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.frequency < other.frequency


# -------------------------------------------------------
# HUFFMAN FUNCTIONS
# -------------------------------------------------------
def calculate_frequency(pixel_array):
    frequency = {}
    for pixel in pixel_array:
        pixel = int(pixel)
        frequency[pixel] = frequency.get(pixel, 0) + 1
    return frequency


def build_huffman_tree(frequency):
    heap = []

    for pixel, freq in frequency.items():
        heapq.heappush(heap, HuffmanNode(pixel, freq))

    if len(heap) == 1:
        root = HuffmanNode(None, list(frequency.values())[0])
        root.left = heapq.heappop(heap)
        return root

    while len(heap) > 1:
        left_node = heapq.heappop(heap)
        right_node = heapq.heappop(heap)

        merged_node = HuffmanNode(None, left_node.frequency + right_node.frequency)
        merged_node.left = left_node
        merged_node.right = right_node

        heapq.heappush(heap, merged_node)

    return heap[0]


def generate_huffman_codes(root, current_code="", codes=None):
    if codes is None:
        codes = {}

    if root is None:
        return codes

    if root.pixel is not None:
        codes[root.pixel] = current_code if current_code != "" else "0"
        return codes

    generate_huffman_codes(root.left, current_code + "0", codes)
    generate_huffman_codes(root.right, current_code + "1", codes)

    return codes


def encode_pixels(pixel_array, codes):
    return "".join(codes[int(pixel)] for pixel in pixel_array)


def decode_bits(encoded_bits, root):
    decoded_pixels = []
    current_node = root

    for bit in encoded_bits:
        current_node = current_node.left if bit == "0" else current_node.right

        if current_node.pixel is not None:
            decoded_pixels.append(current_node.pixel)
            current_node = root

    return np.array(decoded_pixels, dtype=np.uint8)


# -------------------------------------------------------
# ANALYSIS FUNCTIONS
# -------------------------------------------------------
def entropy_value(frequency, total_pixels):
    entropy = 0
    for freq in frequency.values():
        probability = freq / total_pixels
        entropy += probability * np.log2(1 / probability)
    return entropy


def average_code_length(frequency, codes, total_pixels):
    avg = 0
    for pixel, freq in frequency.items():
        probability = freq / total_pixels
        avg += probability * len(codes[pixel])
    return avg


def create_table(frequency, codes, total_pixels, channel):
    data = []

    for pixel, freq in frequency.items():
        data.append({
            "Channel": channel,
            "Pixel Value": pixel,
            "Frequency": freq,
            "Probability": round(freq / total_pixels, 6),
            "Huffman Code": codes[pixel],
            "Code Length": len(codes[pixel])
        })

    return pd.DataFrame(data).sort_values(by="Frequency", ascending=False)


def compress_channel(channel_array, channel_name):
    flat_pixels = channel_array.flatten()
    total_pixels = len(flat_pixels)

    frequency = calculate_frequency(flat_pixels)
    root = build_huffman_tree(frequency)
    codes = generate_huffman_codes(root)

    encoded_bits = encode_pixels(flat_pixels, codes)
    decoded_pixels = decode_bits(encoded_bits, root)
    reconstructed_channel = decoded_pixels.reshape(channel_array.shape)

    table = create_table(frequency, codes, total_pixels, channel_name)

    return {
        "channel": channel_name,
        "frequency": frequency,
        "root": root,
        "codes": codes,
        "encoded_bits": encoded_bits,
        "reconstructed": reconstructed_channel,
        "table": table,
        "entropy": entropy_value(frequency, total_pixels),
        "avg_code_length": average_code_length(frequency, codes, total_pixels),
        "unique_pixels": len(frequency),
        "min_code_length": min(len(code) for code in codes.values()),
        "max_code_length": max(len(code) for code in codes.values()),
        "tree_height": max(len(code) for code in codes.values())
    }


# -------------------------------------------------------
# DOWNLOAD FUNCTIONS
# -------------------------------------------------------
def bits_to_bytes(bit_string):
    padding = 8 - len(bit_string) % 8

    if padding == 8:
        padding = 0

    padded_bits = bit_string + ("0" * padding)
    byte_array = bytearray()

    for i in range(0, len(padded_bits), 8):
        byte_array.append(int(padded_bits[i:i + 8], 2))

    return bytes(byte_array), padding


def create_compressed_file(channel_results, image_shape):
    package = {
        "Project": "Color Image Compression using Huffman Coding",
        "Method": "RGB Channel-wise Huffman Coding",
        "Image Shape": image_shape,
        "Channels": {}
    }

    total_bytes = 0

    for channel, result in channel_results.items():
        compressed_bytes, padding = bits_to_bytes(result["encoded_bits"])
        total_bytes += len(compressed_bytes)

        package["Channels"][channel] = {
            "Compressed Bytes": compressed_bytes,
            "Padding": padding,
            "Codes": result["codes"]
        }

    buffer = io.BytesIO()
    pickle.dump(package, buffer)
    buffer.seek(0)

    return buffer, total_bytes


def image_download_buffer(image_array):
    image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def csv_download_buffer(df):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


# -------------------------------------------------------
# VISUALIZATION FUNCTIONS
# -------------------------------------------------------
def plot_frequency_plotly(df, channel):
    top_df = df.head(20)
    fig = px.bar(
        top_df,
        x="Pixel Value",
        y="Frequency",
        title=f"{channel} Channel Top 20 Pixel Frequencies",
        color="Frequency"
    )
    return fig


def plot_code_length_plotly(df, channel):
    top_df = df.head(20)
    fig = px.bar(
        top_df,
        x="Pixel Value",
        y="Code Length",
        title=f"{channel} Channel Huffman Code Lengths",
        color="Code Length"
    )
    return fig


def plot_rgb_histogram(image_array):
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["red", "green", "blue"]
    labels = ["Red", "Green", "Blue"]

    for i in range(3):
        ax.hist(
            image_array[:, :, i].flatten(),
            bins=256,
            alpha=0.5,
            color=colors[i],
            label=labels[i]
        )

    ax.set_title("RGB Pixel Intensity Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.legend()

    return fig


# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
with st.sidebar:
    st.header("Control Panel")

    uploaded_file = st.file_uploader(
        "Upload Color Image",
        type=["jpg", "jpeg", "png", "bmp"]
    )

    selected_channel = st.selectbox(
        "Select Channel for Analysis",
        ["Red", "Green", "Blue"]
    )

    show_frequency_table = st.checkbox("Show Frequency Table", True)
    show_huffman_table = st.checkbox("Show Huffman Code Table", True)
    show_binary_preview = st.checkbox("Show Binary Preview", True)
    show_charts = st.checkbox("Show Charts", True)

    st.info("This app compresses RGB channels separately and reconstructs the original color image.")


# -------------------------------------------------------
# TABS
# -------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Project Overview",
    "Live Compression",
    "Channel Analysis",
    "DAA Analysis",
    "Downloads",
    "PPT / Viva",
    "About Project"
])


# -------------------------------------------------------
# TAB 1: PROJECT OVERVIEW
# -------------------------------------------------------
with tab1:
    st.subheader("Project Objective")

    st.markdown("""
    <div class='card'>
    The objective of this project is to reduce image file size using Huffman Coding 
    without losing image quality. The system applies Huffman Coding separately on 
    Red, Green, and Blue channels and reconstructs the original color image after decompression.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Mathematical Formula")

    st.latex(r"Compression\ Ratio = \frac{Original\ Size}{Compressed\ Size}")
    st.latex(r"Compression\ Efficiency = \frac{Entropy}{Average\ Code\ Length} \times 100")

    st.subheader("Project Workflow")

    steps = [
        ("Step 1: Image Upload", "The user uploads a color image such as JPG, JPEG, PNG, or BMP."),
        ("Step 2: RGB Channel Separation", "The image is separated into Red, Green, and Blue channels."),
        ("Step 3: Frequency Analysis", "Pixel frequency values are calculated for every RGB channel."),
        ("Step 4: Huffman Tree Construction", "A separate Huffman Tree is built for each RGB channel."),
        ("Step 5: Huffman Code Generation", "Shorter codes are assigned to frequent pixels and longer codes to rare pixels."),
        ("Step 6: Encoding", "RGB pixel values are converted into Huffman binary bitstreams."),
        ("Step 7: Decoding", "Encoded bitstreams are decoded using Huffman Trees."),
        ("Step 8: Color Reconstruction", "Decoded RGB channels are merged to reconstruct the original color image.")
    ]

    for title, desc in steps:
        st.markdown(f"""
        <div class='step-card'>
        <b>{title}</b><br>{desc}
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Architecture Flow")

    st.code("""
Input Color Image
        ↓
RGB Channel Separation
        ↓
Frequency Calculation for R, G, B
        ↓
Huffman Tree Construction
        ↓
Huffman Code Generation
        ↓
RGB Channel Encoding
        ↓
Compressed RGB Data
        ↓
RGB Channel Decoding
        ↓
Merge R + G + B
        ↓
Reconstructed Color Image
""", language="text")


# -------------------------------------------------------
# TAB 2: LIVE COMPRESSION
# -------------------------------------------------------
with tab2:
    st.subheader("Live Color Image Compression")

    if uploaded_file is None:
        st.warning("Please upload an image from the sidebar.")
    else:
        start = time.time()

        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)

        red = image_array[:, :, 0]
        green = image_array[:, :, 1]
        blue = image_array[:, :, 2]

        with st.spinner("Compressing Red, Green, and Blue channels using Huffman Coding..."):
            red_result = compress_channel(red, "Red")
            green_result = compress_channel(green, "Green")
            blue_result = compress_channel(blue, "Blue")

        reconstructed_image = np.stack([
            red_result["reconstructed"],
            green_result["reconstructed"],
            blue_result["reconstructed"]
        ], axis=2)

        end = time.time()
        processing_time = end - start

        channel_results = {
            "Red": red_result,
            "Green": green_result,
            "Blue": blue_result
        }

        total_pixels = image_array.shape[0] * image_array.shape[1] * 3
        original_size_bits = total_pixels * 8

        compressed_size_bits = (
            len(red_result["encoded_bits"]) +
            len(green_result["encoded_bits"]) +
            len(blue_result["encoded_bits"])
        )

        original_size_bytes = original_size_bits / 8
        compressed_size_bytes = compressed_size_bits / 8

        compression_ratio = original_size_bits / compressed_size_bits
        space_saved = ((original_size_bits - compressed_size_bits) / original_size_bits) * 100
        memory_saved = original_size_bytes - compressed_size_bytes

        avg_entropy = (
            red_result["entropy"] +
            green_result["entropy"] +
            blue_result["entropy"]
        ) / 3

        avg_code_length = (
            red_result["avg_code_length"] +
            green_result["avg_code_length"] +
            blue_result["avg_code_length"]
        ) / 3

        compression_efficiency = (avg_entropy / avg_code_length) * 100
        lossless = np.array_equal(image_array, reconstructed_image)

        combined_df = pd.concat([
            red_result["table"],
            green_result["table"],
            blue_result["table"]
        ], ignore_index=True)

        st.session_state["image_array"] = image_array
        st.session_state["reconstructed_image"] = reconstructed_image
        st.session_state["channel_results"] = channel_results
        st.session_state["combined_df"] = combined_df
        st.session_state["image_shape"] = image_array.shape

        col1, col2 = st.columns(2)

        with col1:
            st.image(image_array, caption="Original Color Image", use_container_width=True)

        with col2:
            st.image(reconstructed_image, caption="Reconstructed Color Image", use_container_width=True)

        st.write("---")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original Size", f"{original_size_bytes:.2f} Bytes")
        c2.metric("Compressed Size", f"{compressed_size_bytes:.2f} Bytes")
        c3.metric("Compression Ratio", f"{compression_ratio:.2f} : 1")
        c4.metric("Space Saved", f"{space_saved:.2f}%")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Memory Saved", f"{memory_saved:.2f} Bytes")
        c6.metric("Average Entropy", f"{avg_entropy:.3f}")
        c7.metric("Average Code Length", f"{avg_code_length:.3f} bits")
        c8.metric("Compression Efficiency", f"{compression_efficiency:.2f}%")

        st.metric("Processing Time", f"{processing_time:.3f} seconds")

        if lossless:
            st.markdown("""
            <div class='green-card'>
            <b>Final Result:</b> Color image reconstructed successfully without data loss.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='red-card'>
            <b>Result:</b> Reconstruction mismatch found.
            </div>
            """, unsafe_allow_html=True)

        st.subheader("RGB Histogram")
        st.pyplot(plot_rgb_histogram(image_array))


# -------------------------------------------------------
# TAB 3: CHANNEL ANALYSIS
# -------------------------------------------------------
with tab3:
    st.subheader("Frequency and Huffman Code Analysis")

    if "channel_results" not in st.session_state:
        st.warning("Please upload and compress an image first.")
    else:
        result = st.session_state["channel_results"][selected_channel]
        df = result["table"]

        st.subheader(f"{selected_channel} Channel Statistics")

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Unique Pixels", result["unique_pixels"])
        s2.metric("Tree Height", result["tree_height"])
        s3.metric("Minimum Code Length", result["min_code_length"])
        s4.metric("Maximum Code Length", result["max_code_length"])

        if show_frequency_table:
            st.subheader("Frequency Table")
            st.dataframe(
                df[["Channel", "Pixel Value", "Frequency", "Probability"]],
                use_container_width=True
            )

        if show_huffman_table:
            st.subheader("Huffman Code Table")
            st.dataframe(
                df[["Channel", "Pixel Value", "Frequency", "Huffman Code", "Code Length"]],
                use_container_width=True
            )

        if show_charts:
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(plot_frequency_plotly(df, selected_channel), use_container_width=True)

            with col2:
                st.plotly_chart(plot_code_length_plotly(df, selected_channel), use_container_width=True)

        if show_binary_preview:
            st.subheader("Encoded Binary Preview")
            st.code(result["encoded_bits"][:2000] + " ...", language="text")


# -------------------------------------------------------
# TAB 4: DAA ANALYSIS
# -------------------------------------------------------
with tab4:
    st.subheader("Design and Analysis of Algorithms")

    st.markdown("""
    <div class='card'>
    Huffman Coding is a Greedy Algorithm. It repeatedly selects the two minimum frequency
    nodes using a Min Heap and merges them to build an optimal prefix tree.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Algorithm Type", "Greedy")
    col2.metric("Data Structure", "Min Heap")
    col3.metric("Tree Type", "Binary Tree")
    col4.metric("Encoding Type", "Prefix Code")

    st.subheader("Pseudocode")

    st.code("""
Algorithm: Color Image Compression using Huffman Coding

Input: Color Image I
Output: Compressed RGB data and reconstructed color image

Step 1: Start
Step 2: Read input color image
Step 3: Separate image into Red, Green, and Blue channels
Step 4: For each channel:
            a. Extract pixel values
            b. Calculate frequency of each pixel value
            c. Insert pixel-frequency pairs into Min Heap
            d. While heap has more than one node:
                    i. Remove two minimum frequency nodes
                    ii. Merge them into a new internal node
                    iii. Insert the new node back into heap
            e. Remaining node becomes root of Huffman Tree
            f. Generate Huffman Codes by tree traversal
            g. Encode pixels using Huffman Codes
Step 5: Store compressed RGB bitstreams
Step 6: Decode each channel using its Huffman Tree
Step 7: Merge decoded R, G, and B channels
Step 8: Reconstruct color image
Step 9: Display compression ratio and analysis
Step 10: Stop
""", language="text")

    st.subheader("Complexity Analysis")

    st.latex(r"Time\ Complexity = O(N + n\log n)")
    st.latex(r"Space\ Complexity = O(N+n)")

    st.markdown("""
- **N** = Total number of pixels in all RGB channels  
- **n** = Number of unique pixel values  
- Frequency calculation takes **O(N)** time.  
- Huffman Tree construction using Min Heap takes **O(n log n)** time.  
- Encoding and decoding take **O(N)** time.  
""")


# -------------------------------------------------------
# TAB 5: DOWNLOADS
# -------------------------------------------------------
with tab5:
    st.subheader("Download Results")

    if "channel_results" not in st.session_state:
        st.warning("Please upload and compress an image first.")
    else:
        compressed_file, compressed_byte_size = create_compressed_file(
            st.session_state["channel_results"],
            st.session_state["image_shape"]
        )

        reconstructed_file = image_download_buffer(st.session_state["reconstructed_image"])
        csv_file = csv_download_buffer(st.session_state["combined_df"])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button(
                "Download Compressed RGB File",
                data=compressed_file,
                file_name="huffman_color_compressed.bin",
                mime="application/octet-stream"
            )

        with col2:
            st.download_button(
                "Download Reconstructed Color Image",
                data=reconstructed_file,
                file_name="reconstructed_color_image.png",
                mime="image/png"
            )

        with col3:
            st.download_button(
                "Download Huffman Table CSV",
                data=csv_file,
                file_name="rgb_huffman_table.csv",
                mime="text/csv"
            )


# -------------------------------------------------------
# TAB 6: PPT / VIVA
# -------------------------------------------------------
with tab6:
    st.subheader("PPT / Viva Explanation")

    st.markdown("""
### Project Title
**Image Compression using Huffman Coding**

### Objective
To reduce image file size using Huffman Coding without losing image quality.

### Working
First, the user uploads a color image. The image is divided into Red, Green, and Blue channels.
For each channel, pixel frequency is calculated. Based on frequency values, a Huffman Tree is built.
Frequently occurring pixels get shorter binary codes, and rarely occurring pixels get longer binary codes.
After compression, each channel is decoded and merged again to reconstruct the original color image.

### Formula
**Compression Ratio = Original Size / Compressed Size**

### DAA Concept
This project uses the **Greedy Algorithm** and **Min Heap** data structure.

### Output
The web app displays original image, reconstructed color image, compression ratio,
space saved, memory saved, entropy, average code length, Huffman table, charts, and download options.
""")


# -------------------------------------------------------
# TAB 7: ABOUT PROJECT
# -------------------------------------------------------
with tab7:
    st.subheader("About This Project")

    st.markdown("""
    <div class='card'>
    <b>Project:</b> Image Compression using Huffman Coding<br><br>
    <b>Technology:</b> Python, Streamlit, NumPy, Pandas, PIL, Matplotlib, Plotly<br><br>
    <b>Algorithm:</b> Huffman Coding<br><br>
    <b>DAA Method:</b> Greedy Algorithm<br><br>
    <b>Data Structure:</b> Min Heap / Priority Queue<br><br>
    <b>Output:</b> Color image reconstruction, compression metrics, tables, charts, and downloadable files.
    </div>
    """, unsafe_allow_html=True)
