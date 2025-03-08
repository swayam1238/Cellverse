# Cellverse
Cellverse is an AI-powered assistant designed to enhance cataract surgery procedures by providing real-time video analysis, surgical technique recommendations, and comprehensive feedback. Leveraging advanced machine learning, computer vision, and natural language processing (NLP) technologies, Cellverse aims to support surgeons in achieving optimal patient outcomes.

Features
Surgical Video Analysis: Real-time detection of surgical tools, eye regions, and procedural steps to assist surgeons during operations.
AI-Based Surgical Technique Prediction: Utilizes a Random Forest model to recommend the most suitable surgical approach based on patient data.
Potential Risk Indicator: Assesses and highlights potential risk factors associated with patient complications.
Real-Time Depth Estimation: Employs computer vision techniques to monitor and estimate the positioning of surgical instruments.
Automated Report Generation: Delivers structured feedback and step-by-step guidance using AI-driven NLP capabilities.
Interactive User Interface: Features a Streamlit-powered dashboard for real-time surgical insights and decision-making support.
Technologies Used
Machine Learning & AI: PyTorch, Scikit-Learn (Random Forest), Zero-Shot Learning
Computer Vision: OpenCV, PyTesseract, MediaPipe, YOLO
NLP & Large Language Models: Ollama, OpenCLIP
Deployment & Data Processing: Streamlit (Web App), Pickle (Model Saving), NumPy (Data Processing)
Hardware Acceleration: OpenCL (GPU Optimization)
Getting Started
To explore and utilize the functionalities of Cellverse:

Clone the Repository
git clone https://github.com/swayam1238/Cellverse.git

2.Navigate to the Project Directory:

cd Cellverse

3.Install Dependencies: Ensure you have Python installed. 

4.Run the Application: Launch the Streamlit application with


streamlit run app.py
Usage
Upon running the application:

Video Analysis: Upload surgical videos to receive real-time analysis, including tool detection and depth estimation.
Technique Prediction: Input patient data to obtain AI-based surgical technique recommendations.
Feedback Generation: Receive comprehensive feedback and guidance based on analyzed data.
Contributing
We welcome contributions to enhance Cellverse:

Fork the Repository: Click on the 'Fork' button at the top right corner of this page.
Create a New Branch: Use a descriptive name for your branch.
Commit Your Changes: Provide clear and concise commit messages.
Submit a Pull Request: Explain the changes and improvements you've made.
License
This project is licensed under the MIT License. For more details, refer to the LICENSE file.

Acknowledgements
We extend our gratitude to all contributors and the open-source community for their invaluable support and resources that have made this project possible.
