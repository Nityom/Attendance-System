# Face Recognition Based Attendance System

A smart attendance system using facial recognition, built with Python, OpenCV, and Flask. The system captures faces through a camera, recognizes registered individuals, and maintains attendance records in a SQLite database.

## Features

- Face detection and recognition using dlib
- Web interface to view attendance records
- Automated session-based attendance tracking
- Database storage with SQLite
- Snapshot-based attendance validation

## Prerequisites

Install the required packages:

```sh
pip install -r requirements.txt
```

Required packages:

- dlib
- numpy
- scikit-image
- pandas
- opencv-python
- flask

## Project Structure

```
face-recognition-attendance/
├── app.py                  # Flask web application
├── attendance_system.py    # Core attendance system functionality
├── face_recognition.py     # Face detection and recognition module
├── database.py             # Database operations
├── static/                 # Static web files
│   ├── css/
│   └── js/
├── templates/              # HTML templates
│   ├── index.html
│   └── view_attendance.html
├── data/
│   ├── faces/              # Registered face images
│   ├── features/           # Extracted face features
│   └── attendance.db       # SQLite database
└── requirements.txt
```

## Usage

1. **Register new faces**:
```sh
python register_faces.py
```

2. **Extract features from registered faces**:
```sh
python extract_features.py
```

3. **Start the attendance system**:
```sh
python app.py
```

4. **View attendance records through web interface**:
Then open http://localhost:5000 in your browser.

## How It Works

1. **Face Registration**:
   - Captures face images of individuals through a GUI interface
   - Stores images in the `data/faces/` directory

2. **Feature Extraction**:
   - Processes registered faces to extract 128D face descriptors
   - Stores features in the `data/features/` directory

3. **Attendance Taking**:
   - Starts a timed session (default 2 minutes)
   - Takes snapshots every 10 seconds
   - Requires minimum snapshots for marking attendance

4. **Web Interface**:
   - Provides date-wise attendance records viewing
   - Allows filtering and exporting attendance data

## Database Schema

The system uses SQLite with the following structure:

```sql
CREATE TABLE attendance (
    name TEXT,
    time TEXT,
    date DATE,
    snapshots INTEGER DEFAULT 0,
    status TEXT,
    UNIQUE(name, date)
)
```

## Configuration

The system can be configured by modifying the `config.py` file:

```python
# Sample configuration
FACE_DETECTION_THRESHOLD = 0.6
SESSION_DURATION = 120  # seconds
SNAPSHOT_INTERVAL = 10  # seconds
MIN_SNAPSHOTS_FOR_ATTENDANCE = 3
```

## Troubleshooting

Common issues and solutions:

- **Camera not detected**: Ensure your camera is properly connected and not being used by another application.
- **Face not recognized**: Try registering under better lighting conditions or with multiple angles.
- **dlib installation issues**: Follow the detailed installation guide in the documentation for your specific OS.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

