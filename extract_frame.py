import cv2
from pathlib import Path

# Paths
videos_dir = Path("videos")
frames_dir = Path("frames")

frames_dir.mkdir(exist_ok=True)

# Recursively find all video files
video_files = list(videos_dir.rglob("*.mp4")) + list(videos_dir.rglob("*.mov"))

print(f"Found {len(video_files)} videos.")

for video_file in video_files:
    # Get parent folder (person1 / person2)
    person_folder = video_file.parent.name  
    
    # Create corresponding frames/personX folder
    person_frames_dir = frames_dir / person_folder
    person_frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {video_file} -> {person_frames_dir}")

    cap = cv2.VideoCapture(str(video_file))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Make unique filenames: include video name + frame number
        frame_path = person_frames_dir / f"{video_file.stem}_frame_{frame_count:05d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_file.name}")

print("✅ All videos processed. Frames grouped by person only.")
