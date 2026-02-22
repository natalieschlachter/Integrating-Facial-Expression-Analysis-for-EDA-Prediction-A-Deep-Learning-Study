import cv2
import os
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# video and frames utils
def open_video(video_path):
    """
    Opens a video file and returns the VideoCapture object.

    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    return cap

def get_video_info(folder_path, frames_per_second=5):
    """
    Retrieves and prints video information for all video files in a folder: frame count, FPS, and duration.

    Parameters:
        folder_path (str): Path to the folder containing video files.
        frames_per_second (int): Number of frames per second used for extraction.
    """
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.avi', '.mp4'))]

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        cap = open_video(video_path)
        if not cap:
            continue

        # Get total number of frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get frames per second (FPS)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate video duration
        if fps > 0:
            duration = frame_count / fps
        else:
            print("Error: FPS value is zero, cannot calculate duration.")
            duration = None

        # Calculate total number of extracted frames
        if fps > 0:
            extracted_frame_count = int(duration * frames_per_second)
        else:
            extracted_frame_count = None

        cap.release()

        print(f"Video: {video_file}")
        print(f"Total number of frames: {frame_count}")
        print(f"Frames per second (FPS): {fps}")
        print(f"Duration of the video: {duration} seconds")
        print(f"Total number of extracted frames (at {frames_per_second} fps): {extracted_frame_count}\n")


def save_display_frames(folder_path, target_size=(240, 240)):
    """
    Saves 5 frames from the video file in the given folder, resizes them, converts them to grayscale,
    and saves them in a dedicated folder named 'frames_output'. Also displays the saved frames.

    Parameters:
    - folder_path (str): Path to the folder containing the video file.
    - target_size (tuple): Target size for each saved frame (default is 240x240).
    """
    # Create an output directory in the same location as folder_path
    output_folder = os.path.join(folder_path, 'frames_output')
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of video files in the folder
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.avi', '.mp4'))]

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video file: {video_file}")
            continue

        # Get total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate the interval to ensure 5 frames are evenly spaced
        if total_frames < 5:
            print(f"Video {video_file} has less than 5 frames. Skipping.")
            cap.release()
            continue
        frame_interval = total_frames // 5

        # Frame counter
        frame_count = 0
        saved_frames = 0
        frames_to_display = []

        while saved_frames < 5:
            # Set the video to the desired frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()

            # Break the loop if no frames are returned (end of video)
            if not ret:
                break

            # Resize frame to target size
            resized_frame = cv2.resize(frame, target_size)
            # Convert frame to grayscale
            frame_gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            
            # Save the frame to the output folder
            frame_filename = f"{os.path.splitext(video_file)[0]}_frame_{saved_frames + 1}.png"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame_gray)
            print(f"Saved frame {saved_frames + 1} from {video_file} to {frame_path}")

            # Store the frame for displaying later
            frames_to_display.append(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

            # Increment the saved frame count and frame counter
            saved_frames += 1
            frame_count += frame_interval

        cap.release()

        # Display the saved frames in a row using matplotlib
        if frames_to_display:
            fig, axes = plt.subplots(1, 5, figsize=(20, 5))
            fig.suptitle(f"Frames from {video_file}")
            for i, frame in enumerate(frames_to_display):
                axes[i].imshow(frame)
                axes[i].axis('off')
                axes[i].set_title(f"Frame {i + 1}")
            plt.show()



# csv utils 
def read_csv_data(csv_file_path):
    """
    Reads BVP data from a CSV file.
    
    Parameters:
    - bvp_file_path (str): Path to the BVP CSV file.

    Returns:
    - list: A list of BVP signal values.
    """
    import pandas as pd
    csv_data = pd.read_csv(csv_file_path, header=None)  
    # Return the first column as a list
    return csv_data.iloc[:, 0].tolist()  


def plot_csv_timeseries(folder_path, signal):
    """
    Plots time series and density plots for each CSV file in the given folder.

    Parameters:
    - folder_path (str): Path to the folder containing CSV files.
    - signal (str): Type of signal to plot ('bvp' or 'eda').
    """
    if signal == "bvp":
        print(f"BVP signals are sampled at 64 Hz i.e. 64 signals per second")
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f.startswith('bvp')]
        sampling_rate = 64
    elif signal == "eda":
        print(f"EDA signals are sampled at 4 Hz i.e. 4 signals per second")
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f.startswith('eda')]
        sampling_rate = 4
    else:
        print("Unsupported signal type. Please use 'bvp' or 'eda'.")
        return

    for file in files:
        file_path = os.path.join(folder_path, file)
        data = read_csv_data(file_path)
        num_signals = len(data)
        duration = num_signals / sampling_rate
        
        print(f"File path: {file_path}")
        print(f"Total number of signals: {num_signals}")
        print(f"Number of signals per second: {sampling_rate}; Video Duration: {duration:.2f} seconds")

        # Create subplots for time series and density plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        fig.suptitle(f"Plots for {file}", fontsize=16, fontweight='bold')

        # Time Series Plot
        time = [i / sampling_rate for i in range(num_signals)]
        sns.lineplot(ax=axes[0], x=time, y=data, color='darkred' if signal == 'bvp' else 'blue', linewidth=1)
        axes[0].set_xlabel('Time (seconds)', fontsize=12)
        axes[0].set_ylabel(f'{signal.upper()} Value', fontsize=12)
        axes[0].set_title(f'{signal.upper()} Signals over Time', fontsize=14)
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # Density Plot
        sns.kdeplot(ax=axes[1], data=data, color='darkred' if signal == 'bvp' else 'blue', fill=True, linewidth=0.5, alpha=0.4)
        axes[1].set_xlabel(f'{signal.upper()} Value', fontsize=12)
        axes[1].set_title(f'Density of {signal.upper()} Values', fontsize=14)
        axes[1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def peaks_and_frames(folder_path, signal, threshold=0.5):

    """
    This function extracts frames from video files located in the given folder path, based on significant peaks
    in the corresponding physiological signals (EDA or BVP). Peaks are identified from CSV data files, and the
    frames at the times of these peaks are displayed in rows of 5 figures per row.

    Parameters:
    - folder_path (str): Path to the folder containing video and CSV files.
    - signal (str): Type of signal to analyze ('eda' or 'bvp').
    - threshold (float): Threshold multiplier for detecting significant peaks in the signal.

    """
    # Get the list of video files in the folder
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.avi', '.mp4'))]

    # Set sampling rate based on signal type
    if signal == "eda":
        print(f"EDA peaks and their corresponding frames:")
        sampling_rate = 4
        line_color = 'blue'
    elif signal == "bvp":
        print(f"BVP peaks and their corresponding frames:")
        sampling_rate = 64
        line_color = 'darkred'
    else:
        print("Unsupported signal type. Please use 'eda' or 'bvp'.")
        return

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video file: {video_path}")
            continue

        # Get video frames per second (fps)
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        # Matching csv file 
        identifier = '_'.join(video_file.split('_')[1:]).rsplit('.', 1)[0]
        file_name = f"{signal}_{identifier}.csv"
        
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            print(f"Corresponding CSV file not found for video: {video_file}")
            continue

        # Read the CSV data
        data = read_csv_data(file_path)
        signals = np.array(data)

        # Find peaks in the signals
        peaks, _ = find_peaks(signals, height=np.mean(signals) + np.std(signals) * threshold)

        # Plot signal and highlight peaks
        plt.figure(figsize=(8, 3))
        plt.plot(signals, label=f'{signal.upper()} Signal', color=line_color)
        plt.plot(peaks, signals[peaks], 'x', color='black', label='Peaks')
        plt.title(f'{signal.upper()} Signal with Peaks for {video_file}')
        plt.xlabel('Time (samples)')
        plt.ylabel(f'{signal.upper()} Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

        # Calculate the time (in seconds) for each peak
        times = [peak / sampling_rate for peak in peaks]

        # Display frames in rows with 5 figures per row
        plt.figure(figsize=(20, 10))
        num_frames = len(times)
        for i, time in enumerate(times):
            frame_number = int(time * video_fps)  # Calculate the corresponding frame number
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if ret:
                # Convert frame from BGR to RGB for displaying with Matplotlib
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                plt.subplot((num_frames // 5) + 1, 5, i + 1)
                plt.imshow(frame_rgb)
                plt.axis('off')
                plt.title(f"Time: {time:.2f} seconds")
            else:
                print(f"Failed to extract frame at {time:.2f} seconds from {video_file}.")

        plt.tight_layout()
        plt.show()

        cap.release()



# frames extraction utils
def extract_frames(folder_path, frames_per_second=5, target_size=(240, 240)):
    """
    Extracts frames from all videos in folder_path at a specified rate, resizes them, and converts them to grayscale.
    The frames are not saved to disk but are returned as a list.

    Parameters:
    - folder_path (str): Path to the folder containing video files.
    - frames_per_second (int): Number of frames to extract per second (default is 1).
    - target_size (tuple): Target size for each saved frame (default is 240x240).

    Returns:
    - dict: A dictionary with video file names as keys and lists of frames as values.
    """
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.avi', '.mp4'))]
    extracted_frames = {}

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        cap = open_video(video_path)
        if not cap:
            continue

        # Get frames per second (FPS) of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate the frame interval based on the desired extraction rate
        frame_interval = int(fps / frames_per_second)
        
        # Frame counter
        frame_count = 0
        frames_list = []

        while True:
            ret, frame = cap.read()
            
            # Break the loop if no frames are returned (end of video)
            if not ret:
                break
            
            # Save the frame if it's at the desired interval
            if frame_count % frame_interval == 0:
                # Resize frame to target size
                resized_frame = cv2.resize(frame, target_size)
                # Convert frame to grayscale
                frame_gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                frames_list.append(frame_gray)
                print(f"Extracted frame from {video_file} at count {frame_count}")
            
            # Increment the frame count
            frame_count += 1

        extracted_frames[video_file] = frames_list
        cap.release()

    return extracted_frames

def display_extracted_frames(extracted_frames, frame_limit=5):
    """
    Displays a specified number of frames from the extracted frames dictionary using Matplotlib.
    
    Parameters:
        extracted_frames (dict): Dictionary with video file names as keys and lists of frames as values.
        frame_limit (int): Number of frames to display from each video.
    """
    for video_file, frames in extracted_frames.items():
        print(f"Displaying frames for video: {video_file}")
        for i in range(min(frame_limit, len(frames))):
            frame = frames[i]
            plt.imshow(frame, cmap='gray')  # Since frames are in grayscale
            plt.axis('off')
            plt.show()



# integrating bvp and frames

def extract_frames_with_bvp(folder_path, frames_per_second=5, target_size=(240, 240)):
    """
    Extracts frames from all videos in a folder at a specified rate, resizes them, converts them to grayscale, 
    and synchronizes them with BVP data.
    
    Parameters:
    - folder_path (str): Path to the folder containing video files and corresponding BVP CSV files.
    - frames_per_second (int): Number of frames to extract per second (default is 1).
    - target_size (tuple): Target size for each saved frame (default is 240x240).

    Returns:
    - dict: A dictionary with video file names as keys and tuples of (frames, bvp_signals) as values.
    """
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.avi', '.mp4'))]
    extracted_data = {}

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        # Use the full identifier (e.g., 's2_T1') to match the corresponding BVP file
        identifier = '_'.join(video_file.split('_')[1:]).rsplit('.', 1)[0]
        bvp_file_name = f"bvp_{identifier}.csv"
        bvp_file_path = os.path.join(folder_path, bvp_file_name)
        print(f"video path: {video_file}; file path: {bvp_file_path}")

        # Update logic to correctly match BVP files
        if not os.path.exists(bvp_file_path):
            identifier = '_'.join(video_file.split('_')[1:3])  # Match the 's2_T1' part precisely
            bvp_file_name = f"bvp_{identifier}.csv"
            bvp_file_path = os.path.join(folder_path, bvp_file_name)

        if not os.path.exists(bvp_file_path):
            print(f"BVP file not found for video {video_file}. Expected: {bvp_file_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video file: {video_file}")
            continue

        # Read the BVP signal
        bvp_signal = read_csv_data(bvp_file_path)

        # Get frames per second (FPS) of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate the frame interval based on the desired extraction rate
        frame_interval = int(fps / frames_per_second)
        
        # Frame counter
        frame_count = 0
        frames_list = []
        bvp_list = []

        while True:
            ret, frame = cap.read()
            
            # Break the loop if no frames are returned (end of video)
            if not ret:
                break
            
            # Save the frame if it's at the desired interval
            if frame_count % frame_interval == 0:
                # Resize frame to target size
                resized_frame = cv2.resize(frame, target_size)
                # Convert frame to grayscale
                frame_gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                frames_list.append(frame_gray)

                # Get the corresponding BVP value based on time/frame count
                time_in_seconds = frame_count / fps
                bvp_index = min(int(time_in_seconds * len(bvp_signal) / (cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)), len(bvp_signal) - 1)
                bvp_list.append(bvp_signal[bvp_index])
                print(f"Extracted frame from {video_file} at count {frame_count} with BVP value {bvp_signal[bvp_index]}")
            
            # Increment the frame count
            frame_count += 1

        extracted_data[video_file] = (frames_list, bvp_list)
        cap.release()

    return extracted_data


def display_random_extracted_frames_with_bvp(extracted_data, frame_limit=5):
    """
    Displays a specified number of random extracted frames and corresponding BVP values using Matplotlib.
    
    Parameters:
        extracted_data (dict): Dictionary with video file names as keys and tuples of (frames, bvp_signals) as values.
        frame_limit (int): Number of frames to display for each video.
    """
    for video_file, (frames, bvps) in extracted_data.items():
        print(f"Displaying random frames for video: {video_file}")
        indices = random.sample(range(len(frames)), min(frame_limit, len(frames)))
        for i in indices:
            frame = frames[i]
            bvp_value = bvps[i]
            plt.imshow(frame, cmap='gray')
            plt.title(f"BVP Value: {bvp_value}")
            plt.axis('off')
            plt.show()
