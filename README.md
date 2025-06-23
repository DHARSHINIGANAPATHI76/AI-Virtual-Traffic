# 🧠 AI Virtual Traffic Detection 🚦🚗

A real-time object detection system that identifies vehicles in traffic footage using a pre-trained SSD MobileNet V2 model with TensorFlow and OpenCV. Built for beginners seeking hands-on experience in computer vision, deep learning, and real-world ML deployment.

---

## 📌 Project Highlights

- 🎯 Detects vehicles like cars, buses, trucks, and motorbikes in real-time
- ⚙️ Uses **TensorFlow SSD MobileNet V2** trained on the **COCO** dataset
- 🖼 Processes traffic video (`traffic_video.mp4.mkv`) locally
- 💡 Beginner-friendly code with full setup instructions
- 💻 Runs on Windows using Python 

## ⚙️ Setup Instructions

### ✅ 1. Clone the Repository
git clone https://github.com/DHARSHINIGANAPATHI76/AI-Virtual-Traffic.git
cd AI-Virtual-Traffic
✅ 2. Create & Activate Virtual Environment (Optional)
python -m venv venv
venv\Scripts\activate    # For Windows
✅ 3. Install Dependencies
If you have a requirements.txt file:
pip install -r requirements.txt
Or install manually:

pip install opencv-python tensorflow numpy
🔽 Download the Pretrained Model
To download and extract the SSD MobileNet V2 model (~170 MB), run:

python download_model.py
This will create:
ssd_mobilenet_v2_coco_2018_03_29/
└── frozen_inference_graph.pb
▶️ Run the Project
Once everything is ready, run the detection system:
python ai_virtual_traffic.py
