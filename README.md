#  BGNet — Blood Group Detection System

An AI-powered blood group detection system built as a B.Tech Major Project at SSIPMT, Raipur.

##  Overview
BGNet uses deep learning (ResNet-18) and image processing (OpenCV) to detect blood groups from microscopic blood sample images. It features a full-stack architecture with a React.js frontend, FastAPI backend, and MongoDB database.

##  Tech Stack
| Layer | Technology |
|-------|-----------|
| Deep Learning Model | PyTorch, ResNet-18 |
| Image Processing | OpenCV |
| Backend API | FastAPI (Python) |
| Frontend | React.js |
| Database | MongoDB |

##  Features
- Upload blood sample image → instant blood group prediction
- ResNet-18 fine-tuned on synthetic blood group dataset
- REST API with FastAPI
- Responsive React.js UI
- MongoDB for storing patient records and predictions

##  Project Structure
```
bgnet/
├── backend/        # FastAPI backend + ML model
├── frontend/       # React.js frontend
└── dataset/        # Training data scripts
```

##  Author
**Aditya Kumar Pandey**
 [GitHub](https://github.com/aditya-pandey-dev)
