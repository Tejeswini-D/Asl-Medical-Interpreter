# ASL Medical Interpreter

Real-time ASL interpreter for emergency medical communication.

## Tech Used
- Python
- OpenCV
- MediaPipe
- Scikit-learn

## How to Run
```bash
python live_predict.py
## 📂 Project Structure
├── dataset_capture.py # Record gestures using webcam
├── record_i.py # Record only the "I" gesture
├── extract_landmarks.py # Extract MediaPipe landmarks
├── merge_datasets.py # Merge CSV datasets
├── inspect_csv.py # Check dataset structure
├── train_model.py # Train ML model
├── predict_live.py # Live ASL prediction
├── yes_no.py # Record Yes/No gestures
├── README.md

## 📹 Recording Gestures
To record gestures from your webcam:
```bash
python dataset_capture.py
Notes
Dataset CSV files and trained models are not uploaded to GitHub
Clear lighting and consistent hand position improve accuracy
The model can be expanded with additional ASL signs