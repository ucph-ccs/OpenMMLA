# Indoor Positioning System (IPS)

This pipeline provides indoor positioning using camera-based tracking with AprilTags. It processes video streams to track the position of participants in a room, enabling spatial analysis of group interactions.

## Pipeline Overview

The system uses multiple cameras to track participants wearing AprilTag markers:
- Video capture from multiple cameras (distributed or centralized)
- AprilTag detection to identify and locate markers
- Coordinate transformation between camera views
- Position tracking of participants
- Synchronization of data from multiple cameras
- Real-time visualization of positions

## Usage Instructions