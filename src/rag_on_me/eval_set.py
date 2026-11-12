"""Test questions for retrieval evaluation.

Each entry lists the source(s) that SHOULD appear in the retrieved chunks.
Source names must match the store exactly: the repo name for GitHub docs,
the file name for PDFs.
"""

EVAL_SET = [
    {
        "question": "Has Abir worked on medical imaging?",
        "expected": [
            "ChestXray-Image-classification-using-PyTorch",
            "Human-BrainMRI-image-segmentation",
            "Eye-disease-classification-with-PyTorch",
        ],
    },
    {
        "question": "Has Abir used XGBoost?",
        "expected": [
            "telco-churn-prediction",
            "DoorDash-delivery-time-prediction-",
        ],
    },
    {
        "question": "What is the SlotFinder system?",
        "expected": ["SlotFinder_A_Spatio-temporal_based_Car_Parking_System.pdf"],
    },
    {
        "question": "What was Abir's MSc thesis about?",
        "expected": [
            "Maximizing Auditory Cortex Output Optimal.pdf",
            "Maximizing-Auditory-cortex-Output-Optimal-Source-Detector-Placement-on-the-scalp",
        ],
    },
    {
        "question": "Which project predicts how long food deliveries take?",
        "expected": ["DoorDash-delivery-time-prediction-"],
    },
    {
        "question": "Has Abir built anything for a bank's customers?",
        "expected": ["N26-Income-and-Expences-prediction-"],
    },
    {
        "question": "Has Abir deployed models with FastAPI or Docker?",
        "expected": [
            "CV_Md_Abir__AkerSolutions.pdf",
            "ChestXray-Image-classification-using-PyTorch",
            "telco-churn-prediction",
        ],
    },
    {
        "question": "What time series forecasting work has Abir done?",
        "expected": [
            "Time-Series-Forecasting-Energy-consumption-",
            "CV_Md_Abir__AkerSolutions.pdf",
            "SlotFinder_A_Spatio-temporal_based_Car_Parking_System.pdf",
        ],
    },
    {
        "question": "What experience does Abir have with AI agents?",
        "expected": ["aboutme.md", "CV_Md_Abir__AkerSolutions.pdf"],
    },
    {
        "question": "How was the eye infection classifier built?",
        "expected": ["Eye-disease-classification-with-PyTorch"],
    },
    {
        "question": "Who is Abir?",
        "expected": ["aboutme.md"],
    },
]
