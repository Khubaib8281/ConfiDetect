import tkinter as tk
from tkinter import Label, Button
import cv2
import os

os.path.join(os.path.dirname(__file__), "mediapipe", "modules", "pose_landmark", "pose_landmark_cpu.binarypb")

import mediapipe as mp
# print(mp.__path__)
pose = mp.solutions.pose.Pose(model_complexity=1)
from PIL import Image, ImageTk
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import Counter
import numpy as np
from tkinter import messagebox
import pandas as pd
import time
import math
import sys

# MediaPipe initialization
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
# pose = mp_pose.Pose()
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# App state
running = False
paused = False
cap = None
frame_data = []
fps_list = []

# Analysis data
confidence_states = []
posture_states = []
head_directions = []
arm_positions = []
numerical_data = []

def analyze_frame(image_rgb, results_pose, results_face):
    global confidence_states, posture_states, head_directions, arm_positions

    height, width, _ = image_rgb.shape
    confidence = "Neutral"        
    posture = "Upright"
    head_dir = "Center"
    arms = "Neutral"

    if results_pose.pose_landmarks:
        landmarks = results_pose.pose_landmarks.landmark

        # Confidence: Based on head (chin) and eyes relative to shoulders
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]


        # Posture: vertical shoulder alignment
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        if shoulder_diff < 0.03:
            posture = "Upright"
        elif shoulder_diff > 0.08:
            posture = "Slouched"
        else:
            posture = "Stiff"
        posture_states.append(posture)

        # Head orientation
        if results_face.multi_face_landmarks:
            face_landmarks = results_face.multi_face_landmarks[0].landmark
            left_eye = face_landmarks[33]
            right_eye = face_landmarks[263]
            nose = face_landmarks[1]

            eye_center_x = (left_eye.x + right_eye.x) / 2

            if nose.x < eye_center_x - 0.03:
                head_dir = "Looking Right"
            elif nose.x > eye_center_x + 0.03:
                head_dir = "Looking Left"
            else:
                head_dir = "Looking Straight"
        head_directions.append(head_dir)


        l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Calculations
        eye_avg_y = (left_eye.y + right_eye.y) / 2
        shoulder_avg_y = (l_shoulder.y + r_shoulder.y) / 2
        hip_avg_y = (l_hip.y + r_hip.y) / 2
        eye_shoulder_y_ratio = eye_avg_y - shoulder_avg_y
        shoulder_y_diff = abs(l_shoulder.y - r_shoulder.y)
        wrist_distance_x = abs(r_wrist.x - l_wrist.x)
        shoulder_span = abs(r_shoulder.x - l_shoulder.x)
        wrist_shoulder_ratio = wrist_distance_x / (shoulder_span + 1e-5)
        nose_eye_center_offset_x = nose.x - ((left_eye.x + right_eye.x) / 2)

        wrist_distance = abs(r_wrist.x - l_wrist.x)
        shoulder_distance = abs(r_shoulder.x - l_shoulder.x)

        # Arms

        # Calculate wrist distance (actual Euclidean distance for better accuracy)
        wrist_distance = math.dist(
            [l_wrist.x, l_wrist.y],
            [r_wrist.x, r_wrist.y]
        )

        # Shoulder span (baseline for body width)
        shoulder_span = math.dist(
            [left_shoulder.x, left_shoulder.y],
            [right_shoulder.x, right_shoulder.y]
        )

        # Normalized ratio of wrist-to-wrist distance to shoulder span
        arm_extension_ratio = wrist_distance / shoulder_span if shoulder_span != 0 else 0

        # Logic
        if arm_extension_ratio > 1.5:
            arms = "Open Arms"
        elif arm_extension_ratio < 1.1:
            arms = "Closed Arms"
        else:
            arms = "Partially Open"

        arm_positions.append(arms)


        # Normalize height difference based on shoulder width

# Shoulder and hip width
        shoulder_width = math.dist([left_shoulder.x, left_shoulder.y], [right_shoulder.x, right_shoulder.y])
        hip_width = math.dist([l_hip.x, l_hip.y], [r_hip.x, r_hip.y])

        # Body openness score (posture)
        posture_score = shoulder_width / hip_width if hip_width != 0 else 0

# Head height relative to body (uprightness)
        eye_avg_y = (left_eye.y + right_eye.y) / 2
        eye_avg_x = (left_eye.x + right_eye.x) / 2
        shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_avg_y = (l_hip.y + r_hip.y) / 2
        eye_center_x = (left_eye.x + right_eye.x) / 2

        torso_length = abs(hip_avg_y - shoulder_avg_y)
        uprightness_score = abs(shoulder_avg_y - eye_avg_y) / torso_length if torso_length != 0 else 0

        # Chin-up score (head tilt upward = confident)
        nose_to_eye = math.dist([nose.x, nose.y], [eye_avg_x, eye_avg_y])
        chin_up_score = nose_to_eye / torso_length if torso_length != 0 else 0

        shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        eye_shoulder_ratio = (shoulder_avg_y - eye_avg_y) / shoulder_width
        shoulder_hip_ratio = (hip_avg_y - shoulder_avg_y) / shoulder_width
        shoulder_slope = abs(right_shoulder.y - left_shoulder.y)
        wrist_distance = abs(r_wrist.x - l_wrist.x)
        wrist_ratio = wrist_distance / shoulder_width
        head_tilt = abs(nose.x - eye_center_x)
        eye_dx = right_eye.x - left_eye.x
        eye_dy = right_eye.y - left_eye.y
        head_tilt_angle = math.degrees(math.atan2(eye_dy, eye_dx)) # degrees
        eye_distance = math.hypot(right_eye.x - left_eye.x, right_eye.y - left_eye.y)
        eye_distance_ratio = eye_distance / shoulder_span

        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        hip_center_x = (l_hip.x + r_hip.x) / 2
        spine_angle = math.degrees(math.atan2(
            (r_hip.y + l_hip.y) / 2 - (right_shoulder.y + left_shoulder.y) / 2,
            hip_center_x - shoulder_center_x
        ))

        body_lean_x = shoulder_center_x - hip_center_x


        if (eye_shoulder_ratio > 0.9 and
            shoulder_slope < 0.05 and
            wrist_ratio > 1.2 and
            head_tilt < 0.05):
            confidence = "Confident"
        elif (eye_shoulder_ratio < 0.6 or
            shoulder_slope > 0.1 or
            wrist_ratio < 0.8):
            confidence = "Low"
        else:
            confidence = "Neutral"


        confidence_states.append(confidence)

        numerical_data.append({
            "eye_shoulder_y_ratio": eye_shoulder_y_ratio,
            "shoulder_y_diff": shoulder_y_diff,
            "wrist_distance_x": wrist_distance_x,
            "wrist_shoulder_ratio": wrist_shoulder_ratio,
            "nose_eye_center_offset_x": nose_eye_center_offset_x,
            "shoulder_span": shoulder_span,
            "hip_shoulder_y_diff": hip_avg_y - shoulder_avg_y,
            "body_lean_x": body_lean_x,
            "shoulder_center_x": shoulder_center_x,
            "hip_center_x": hip_center_x,
            "spine_angle": spine_angle,
            "eye_distance": eye_distance,
            "head_tilt_angle": head_tilt_angle,
            "eye_distance_ratio": eye_distance_ratio,
            "shoulder_slope": shoulder_slope,
            "head_direction": head_directions[-1],       # categorical
            "arm_position": arm_positions[-1],           # categorical
            "posture": posture_states[-1],               # categorical
            "confidence_label": confidence               # categorical
        })



    return confidence, posture, head_dir, arms

def video_loop(label_video, label_info):
    global cap, running, paused, frame_data, fps_list

    prev_time = 0
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)

    while running:
        if paused:
            time.sleep(0.1)    
            continue

        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_list.append(fps)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(image_rgb)
        results_face = face_mesh.process(image_rgb)
        confidence, posture, head_dir, arms = analyze_frame(image_rgb, results_pose, results_face)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(image_bgr, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                mp_drawing.draw_landmarks(image_bgr, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
        frame_data.append((confidence, posture))

        # Update GUI info
        info_text = f"FPS: {fps:.1f} | Confidence: {confidence} | Posture: {posture} | Head: {head_dir} | Arms: {arms}"
        label_info.config(text=info_text)

        # Update video frame
        img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        label_video.imgtk = imgtk
        label_video.configure(image=imgtk)

def start_recording(label_video, label_info):
    global cap, running, paused
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
        paused = False
        threading.Thread(target=video_loop, args=(label_video, label_info)).start()
    elif paused:
        paused = False

def pause_recording():
    global paused
    paused = True

def stop_recording():
    global cap, running, paused
    running = False
    paused = False
    if cap:
        cap.release()

def reset_session(label_info):
    global frame_data, fps_list, confidence_states, posture_states, head_directions, arm_positions
    frame_data.clear()
    fps_list.clear()
    confidence_states.clear()
    posture_states.clear()
    head_directions.clear()
    arm_positions.clear()
    label_info.config(text="Session reset. Ready to record.")

def save_numerical_csv():
    df = pd.DataFrame(numerical_data)
    df.to_csv("numerical_features.csv", index=False)
    # print("Numerical features saved as 'numerical_features.csv'")
    messagebox.showinfo("Success", "Numerical features saved as 'numerical_features.csv'")

def show_results(root):
    global frame_data, confidence_states, posture_states, head_directions, arm_positions

    summary_frame = tk.Toplevel(root, bg="black")
    summary_frame.attributes("-fullscreen", True)
    summary_frame.title("Session Summary")
    btn_back_to_main = tk.Button(summary_frame, text="Back to Main", command=summary_frame.destroy, bg="red", fg="white", font=("Helvetica", 12))
    btn_back_to_main.pack(pady=5)

    most_conf = max(set(confidence_states), key=confidence_states.count)
    most_posture = max(set(posture_states), key=posture_states.count)
    most_head_dir = max(set(head_directions), key=head_directions.count)
    most_arm_positions = max(set(arm_positions), key=arm_positions.count)

    label_summary = Label(summary_frame, text=f"Most Common Confidence: {most_conf}\nMost Common Posture: {most_posture}\nMost Common Head Direction: {most_head_dir}\nMost Common Arm Position: {most_arm_positions}", 
                          font=("Helvetica", 16), fg="White", bg="black")
    label_summary.pack(pady=5)

    # Plotting
    conf_count = Counter(confidence_states)
    post_count = Counter(posture_states)
    head_count = Counter(head_directions)
    arm_count = Counter(arm_positions)

    categories = list(set(conf_count) | set(post_count) | set(head_count) | set(arm_count))

    # --- Prepare Data ---
    conf_vals = [conf_count.get(cat, 0) for cat in categories]
    post_vals = [post_count.get(cat, 0) for cat in categories]
    head_vals = [head_count.get(cat, 0) for cat in categories]
    arm_vals = [arm_count.get(cat, 0) for cat in categories]

    # Scoring logic
    score = 0
    total = len(confidence_states)

    for i in range(total):
        c = 1 if confidence_states[i] == "Confident" else 0.5 if confidence_states[i] == "Neutral" else 0
        p = 1 if posture_states[i] == "Upright" else 0.5 if posture_states[i] == "Neutral" else 0
        h = 1 if head_directions[i] == "Looking Straight" else 0.75  # Less harsh
        a = 1 if arm_positions[i] == "Open" else 0.5  # Partial score
        score += (c + p + h + a) / 4

    final_percentage = (score / total) * 100

    # Optional boost for dominant confidence
    if confidence_states.count("Confident") / total >= 0.6:
        final_percentage += 5

    # Cap between 0 and 100
    final_percentage = min(final_percentage, 100)

    label_score = tk.Label(
    summary_frame,
    text=f"🧠 Final Session Score: {final_percentage:.2f}%",
    font=("Helvetica", 16),
    fg="lightgreen",
    bg="black"
)
    label_score.pack(pady=(5, 5))

    # --- Bar Plot ---
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()  # Easier indexing

    # Plot 1: Confidence
    conf_count = Counter(confidence_states)
    axs[0].bar(conf_count.keys(), conf_count.values(), color='skyblue')
    axs[0].set_title("Confidence")
    axs[0].set_ylabel("Count")
    axs[0].tick_params(axis='x', rotation=15)

    # Plot 2: Posture
    post_count = Counter(posture_states)
    axs[1].bar(post_count.keys(), post_count.values(), color='salmon')
    axs[1].set_title("Posture")
    axs[1].set_ylabel("Count")
    axs[1].tick_params(axis='x', rotation=15)

    # Plot 3: Head Direction
    head_count = Counter(head_directions)
    axs[2].bar(head_count.keys(), head_count.values(), color='lightgreen')
    axs[2].set_title("Head Direction")
    axs[2].set_ylabel("Count")
    axs[2].tick_params(axis='x', rotation=15)
    
    # Plot 4: Arm Position
    arm_count = Counter(arm_positions)
    axs[3].bar(arm_count.keys(), arm_count.values(), color='plum')
    axs[3].set_title("Arm Position")
    axs[3].set_ylabel("Count")
    axs[3].tick_params(axis='x', rotation=15)        

    plt.tight_layout()


    # --- Embed plot into Tkinter ---
    canvas = FigureCanvasTkAgg(fig, master=summary_frame)
    canvas.draw()

    def download_plot():
        fig.savefig("session_summary_plot.png")  
        messagebox.showinfo("Success", f"Plot downloaded successfully as:\nsession_summary_plot.png")
        # print("Plot saved as 'session_summary_plot.png'")

    btn_download = tk.Button(summary_frame, text="Download Plot", command =download_plot, bg="green", fg="white", font=("Helvetica", 12))   
    btn_download.pack(pady=10)

    btn_save_csv = tk.Button(summary_frame, text="Download Numerical CSV", command=save_numerical_csv, bg="orange", fg="white", font=("Helvetica", 12))
    btn_save_csv.pack(pady=5)
    


    canvas.get_tk_widget().pack()

def exit_app():
    global cap, running, paused
    running = False
    paused = False
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    sys.exit()


def build_gui():
    root = tk.Tk()
    root.attributes('-fullscreen', True)
    root.title("Human Confidence & Posture Monitor")
    root.configure(bg="black")

    label_video = Label(root, bg="black")
    label_video.pack()

    label_info = Label(root, text="Initializing...", font=("Helvetica", 14), fg="green", bg="black")
    label_info.pack(pady=10)

    btn_frame = tk.Frame(root, bg="black")
    btn_frame.pack(pady=20)

    Button(btn_frame, text="Start Recording", command=lambda: start_recording(label_video, label_info),
           width=20, bg="green", fg="white", font=("Helvetica", 12, "bold")
).grid(row=0, column=0, padx=10)

    Button(btn_frame, text="Pause Recording", command=pause_recording,
           width=20, bg="orange", fg="white", font=("Helvetica", 12, "bold")
).grid(row=0, column=1, padx=10)    

    Button(btn_frame, text="Stop Recording", command=stop_recording,
           width=20, bg="red", fg="white", font=("Helvetica", 12, "bold")
).grid(row=0, column=2, padx=10)

    Button(btn_frame, text="Show Results", command=lambda: show_results(root),
           width=20, bg="blue", fg="white", font=("Helvetica", 12, "bold")
).grid(row=0, column=3, padx=10)

    Button(btn_frame, text="Reset", command=lambda: reset_session(label_info),
       width=20, bg="purple", fg="white", font=("Helvetica", 12, "bold")
).grid(row=0, column=4, padx=10)

    Button(btn_frame, text="Exit", command=lambda: exit_app(),
       width=20, bg="black", fg="white", font=("Helvetica", 12, "bold")).grid(row=0, column=5, padx=10)


    Label(root, text="Developed by Muhammad Khubaib Ahmad", font=("Helvetica", 12), fg="gray", bg="black").pack(side="bottom", pady=10)

    root.mainloop()

build_gui()   
