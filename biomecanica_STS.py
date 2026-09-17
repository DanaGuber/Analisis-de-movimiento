"""
=============================================================================
ANÁLISIS BIOMECÁNICO STS — Interfaz unificada
=============================================================================
Pestaña PROCESAR: ingresás video derecho e izquierdo (con su distancia
hombro-cadera en cm cada uno) y procesás ambos en secuencia.
Pestaña VISUALIZAR: los resultados ya están cargados. Elegís lado
(der/izq) y tipo de video (trackeado/skeleton). Todo automático.

Requisitos:
    pip install opencv-python mediapipe pandas numpy matplotlib scipy pillow reportlab
=============================================================================
"""

import tkinter as tk  #interfaz gráfica 
from tkinter import ttk, filedialog, messagebox
import threading
import os
import math

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

from PIL import Image, ImageTk, ImageDraw
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import savgol_filter, medfilt

# PDF
import io
import tempfile
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage,
                                 HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# =============================================================================
# COLORES
# =============================================================================

C_BG      = "#E8E8E8"#"#1a1a2e"
C_PANEL   = "#DCDCDC"#"#16213e"
C_CARD    = "#F0F0F0"#"#0f3460"
C_ACCENT  = "#de314d"#"#e94560"
C_ACCENT2 = "#5b07de"#"#533483"
C_TEXT    = "#0a0a0a"#"#eaeaea"
C_SUB     = "#0a0a0a"#"#aaaaaa"
C_GREEN   = "#0dfcc8"#"#4ecca3"
C_YELLOW  = "#fc9f05"#"#f5a623"
C_RED     = "#f20c32"#"#e94560"

# =============================================================================
# NOMBRES LANDMARKS
# =============================================================================

LANDMARK_NAMES = [
    'nariz','ojo_izq_int','ojo_izq','ojo_izq_ext',
    'ojo_der_int','ojo_der','ojo_der_ext',
    'oreja_izq','oreja_der','boca_izq','boca_der',
    'hombro_izq','hombro_der','codo_izq','codo_der',
    'muneca_izq','muneca_der','menique_izq','menique_der',
    'indice_izq','indice_der','pulgar_izq','pulgar_der',
    'cadera_izq','cadera_der','rodilla_izq','rodilla_der',
    'tobillo_izq','tobillo_der','talon_izq','talon_der',
    'pie_izq','pie_der'
]

# =============================================================================
# PROCESAMIENTO — MediaPipe  (sin cambios respecto al original)
# =============================================================================

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

def line_intersection(p1, p2, p3, p4):
    x1,y1=p1; x2,y2=p2; x3,y3=p3; x4,y4=p4
    d = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(d) < 1e-9:
        return None
    px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/d
    py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/d
    return [px, py]

def draw_angle_arc(img, a, b, c, r, color, thickness=2):
    def adeg(p1,p2): return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))
    bx = (int(b[0]),int(b[1]))
    a1, a2 = adeg(b,a), adeg(b,c)
    s, e = sorted([a1,a2])
    if e-s > 180: s, e = e, s+360
    cv2.ellipse(img, bx, (r,r), 0, s, e, color, thickness)

def draw_joint(img, pt, color=(255,255,255), r=6):
    cv2.circle(img, (int(pt[0]),int(pt[1])), r, color, -1)
    cv2.circle(img, (int(pt[0]),int(pt[1])), r+2, (0,0,0), 1)

def put_angle_text(img, angle, pt, offset=(0,0), color=(255,255,255)):
    pos = (int(pt[0])+offset[0], int(pt[1])+offset[1])
    text = f"{int(round(angle))}°"
    cv2.putText(img, text, (pos[0]+1,pos[1]+1), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(img, text,  pos,               cv2.FONT_HERSHEY_SIMPLEX, 0.65,color,  2,cv2.LINE_AA)


def procesar_video(input_path, hombro_cadera_cm, log_fn, progress_fn):
    base_name  = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.dirname(input_path)

    if "_izq" in base_name:
        lado = "izq"
    elif "_der" in base_name:
        lado = "der"
    else:
        raise ValueError(f"El archivo '{base_name}' debe contener '_izq' o '_der' en el nombre.")

    track_path    = os.path.join(output_dir, f"{base_name}_track.mp4")
    skeleton_path = os.path.join(output_dir, f"{base_name}_skeleton.mp4")
    csv_path      = os.path.join(output_dir, f"{base_name}.csv")

    log_fn(f"[{lado.upper()}] Iniciando procesamiento…")
    log_fn(f"[{lado.upper()}] Conversión a cm: {'Sí → ' + str(hombro_cadera_cm) + ' cm' if hombro_cadera_cm else 'No (normalizado)'}")

    ALPHA = 0.7
    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(model_complexity=2, smooth_landmarks=True,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap    = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps_v  = cap.get(cv2.CAP_PROP_FPS) or 30
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    out       = cv2.VideoWriter(track_path,   fourcc, fps_v, (W,H))
    out_black = cv2.VideoWriter(skeleton_path,fourcc, fps_v, (W,H))

    if lado == "izq":
        ids = dict(K=25,A=27,H=23,S=11,E=13,W=15,EAR=7, EYE=3, HEEL=29,FOOT=31)
    else:
        ids = dict(K=26,A=28,H=24,S=12,E=14,W=16,EAR=8, EYE=6, HEEL=30,FOOT=32)

    prev_sm    = {}
    scale_cm   = None
    scale_done = False
    data_all   = []
    frame_idx  = 0

    def sm(name, p):
        if name in prev_sm:
            p = [ALPHA*prev_sm[name][0]+(1-ALPHA)*p[0],
                 ALPHA*prev_sm[name][1]+(1-ALPHA)*p[1]]
        prev_sm[name] = p
        return p

    def n2px(p): return (int(p[0]*W), int(p[1]*H))
    def fp(p):   return (p[0]*W, p[1]*H)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        black = np.zeros((H,W,3), dtype=np.uint8)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            if hombro_cadera_cm and not scale_done:
                try:
                    hx,hy = lm[ids["S"]].x, lm[ids["S"]].y
                    cx,cy = lm[ids["H"]].x, lm[ids["H"]].y
                    d = math.sqrt((hx-cx)**2+(hy-cy)**2)
                    if d > 0.01:
                        scale_cm   = hombro_cadera_cm / d
                        scale_done = True
                        log_fn(f"[{lado.upper()}] Escala: {scale_cm:.2f} cm/unidad-norm")
                except Exception as ex:
                    log_fn(f"[{lado.upper()}] Advertencia escala: {ex}")

            data = {
                "frame": frame_idx, "lado": lado,
                "hombro_cadera_ref_cm": hombro_cadera_cm if hombro_cadera_cm else np.nan,
                "angulo_tobillo": np.nan, "angulo_rodilla": np.nan,
                "angulo_cadera":  np.nan, "angulo_brazo":   np.nan,
                "angulo_cuello_vertical": np.nan,
                "angulo_cabeza_cuello":   np.nan,
            }

            try:
                rod  = sm("rod",  [lm[ids["K"]].x,    lm[ids["K"]].y])
                tob  = sm("tob",  [lm[ids["A"]].x,    lm[ids["A"]].y])
                cad  = sm("cad",  [lm[ids["H"]].x,    lm[ids["H"]].y])
                hom  = sm("hom",  [lm[ids["S"]].x,    lm[ids["S"]].y])
                cod  = sm("cod",  [lm[ids["E"]].x,    lm[ids["E"]].y])
                mun  = sm("mun",  [lm[ids["W"]].x,    lm[ids["W"]].y])
                ear  = sm("ear",  [lm[ids["EAR"]].x,  lm[ids["EAR"]].y])
                eye  = sm("eye",  [lm[ids["EYE"]].x,  lm[ids["EYE"]].y])
                heel = sm("heel", [lm[ids["HEEL"]].x, lm[ids["HEEL"]].y])
                foot = sm("foot", [lm[ids["FOOT"]].x, lm[ids["FOOT"]].y])

                tob_eje = line_intersection(rod, tob, foot, heel)

                if tob_eje and 0<=tob_eje[0]<=1 and 0<=tob_eje[1]<=1:
                    ang_t = calculate_angle(rod, tob_eje, foot)
                    ang_r = calculate_angle(tob_eje, rod, cad)
                    ang_c = calculate_angle(rod, cad, hom)
                    ang_b = calculate_angle(cod, hom, cad)

                    hom_a  = np.array(hom); ear_a = np.array(ear)
                    vec_he = ear_a - hom_a
                    vec_v  = np.array([0.0, -1.0])
                    cos_cv = np.dot(vec_he, vec_v) / (np.linalg.norm(vec_he) + 1e-9)
                    ang_cv = float(np.degrees(np.arccos(np.clip(cos_cv,-1,1))))
                    ang_hc = calculate_angle(hom, ear, eye)

                    data.update({
                        "angulo_tobillo": ang_t, "angulo_rodilla": ang_r,
                        "angulo_cadera":  ang_c, "angulo_brazo":   ang_b,
                        "angulo_cuello_vertical": ang_cv,
                        "angulo_cabeza_cuello":   ang_hc,
                    })

                    p_rod=n2px(rod); p_tob=n2px(tob_eje); p_tob_r=n2px(tob)
                    p_cad=n2px(cad); p_hom=n2px(hom);     p_cod=n2px(cod)
                    p_mun=n2px(mun); p_ear=n2px(ear);     p_foot=n2px(foot)
                    p_heel=n2px(heel)

                    for img in [frame, black]:
                        cv2.line(img,p_rod,  p_tob_r,(200,200,200),2)
                        cv2.line(img,p_tob_r,p_tob,  (0,255,120),  2)
                        cv2.line(img,p_tob,  p_foot,  (200,200,200),2)
                        cv2.line(img,p_rod,  p_cad,   (200,200,200),2)
                        cv2.line(img,p_cad,  p_hom,   (200,200,200),2)
                        cv2.line(img,p_hom,  p_cod,   (200,200,200),2)
                        cv2.line(img,p_cod,  p_mun,   (180,180,180),2)
                        cv2.line(img,p_hom,  p_ear,   (200,200,200),2)
                        cv2.line(img,p_heel, p_foot,  (150,150,150),2)

                        for pt, col in [
                            (p_rod,(0,255,255)),(p_tob,(0,255,0)),(p_cad,(0,220,255)),
                            (p_hom,(255,165,0)),(p_cod,(200,100,255)),(p_mun,(150,150,255)),
                            (p_ear,(255,255,0)),
                        ]:
                            draw_joint(img, pt, col)

                        draw_angle_arc(img,fp(rod),    fp(tob_eje),fp(foot),22,(0,0,255),  2)
                        draw_angle_arc(img,fp(tob_eje),fp(rod),    fp(cad), 42,(255,0,200),2)
                        draw_angle_arc(img,fp(rod),    fp(cad),    fp(hom), 42,(0,255,255),2)
                        draw_angle_arc(img,fp(cod),    fp(hom),    fp(cad), 36,(255,165,0),2)

                        p_vert = (hom[0]*W, (hom[1]-0.15)*H)
                        draw_angle_arc(img,p_vert,fp(hom),fp(ear),30,(0,200,255),2)
                        cv2.line(img,p_hom,(p_hom[0],max(0,p_hom[1]-40)),(0,200,255),1,cv2.LINE_AA)
                        draw_angle_arc(img,fp(hom),fp(ear),fp(eye),22,(255,255,0),2)

                        put_angle_text(img,ang_t, p_tob,  (-55,-10),(100,255,100))
                        put_angle_text(img,ang_r, p_rod,  (-60, 10),(255,0,200))
                        put_angle_text(img,ang_c, p_cad,  (-60,-20),(0,255,255))
                        put_angle_text(img,ang_b, p_hom,  (-50, 25),(255,165,0))
                        put_angle_text(img,ang_cv,p_hom,  ( 10,-15),(0,200,255))
                        put_angle_text(img,ang_hc,p_ear,  (-10,-28),(255,255,0))

            except:
                pass

            for i, landmark in enumerate(lm):
                nm = LANDMARK_NAMES[i]
                data[f"{nm}_x_norm"]     = landmark.x
                data[f"{nm}_y_norm"]     = 1.0 - landmark.y
                data[f"{nm}_z_norm"]     = landmark.z
                data[f"{nm}_visibility"] = landmark.visibility
                if scale_cm:
                    data[f"{nm}_x_cm"] = landmark.x * scale_cm
                    data[f"{nm}_y_cm"] = (1.0-landmark.y)*scale_cm
                    data[f"{nm}_z_cm"] = landmark.z * scale_cm

            def seg_len(i,j):
                dx=lm[i].x-lm[j].x; dy=lm[i].y-lm[j].y
                return math.sqrt(dx*dx+dy*dy)
            segs_def = {
                "seg_hombro_codo":    (ids["S"],ids["E"]),
                "seg_codo_muneca":    (ids["E"],ids["W"]),
                "seg_cadera_rodilla": (ids["H"],ids["K"]),
                "seg_rodilla_tobillo":(ids["K"],ids["A"]),
                "seg_tobillo_pie":    (ids["A"],ids["FOOT"]),
                "seg_hombro_cadera":  (ids["S"],ids["H"]),
            }
            for sn,(si,sj) in segs_def.items():
                try:
                    d = seg_len(si,sj)
                    data[f"{sn}_norm"] = d
                    if scale_cm: data[f"{sn}_cm"] = d*scale_cm
                except:
                    data[f"{sn}_norm"] = np.nan

            mp_drawing.draw_landmarks(frame,res.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(200,200,200),thickness=1,circle_radius=2),
                mp_drawing.DrawingSpec(color=(80,80,80),   thickness=1))
            mp_drawing.draw_landmarks(black,res.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(180,180,180),thickness=1,circle_radius=2),
                mp_drawing.DrawingSpec(color=(60,60,60),   thickness=1))

            for li,line in enumerate([f"Frame: {frame_idx}", f"Lado: {lado.upper()}"]):
                for img in [frame, black]:
                    cv2.putText(img,line,(12,28+li*22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),3,cv2.LINE_AA)
                    cv2.putText(img,line,(12,28+li*22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(220,220,220),1,cv2.LINE_AA)

            data_all.append(data)

        out.write(frame)
        out_black.write(black)
        frame_idx += 1
        progress_fn(min(99, int(frame_idx/total*100)))

    cap.release(); out.release(); out_black.release(); pose.close()

    df = pd.DataFrame(data_all)
    meta   = ["frame","lado","hombro_cadera_ref_cm"]
    angles = [c for c in df.columns if c.startswith("angulo_")]
    segs_c = [c for c in df.columns if c.startswith("seg_")]
    lms    = [c for c in df.columns if c not in meta+angles+segs_c]
    df     = df[meta+angles+segs_c+lms]
    df.to_csv(csv_path, index=False)

    progress_fn(100)
    log_fn(f"[{lado.upper()}] ✔ {frame_idx} frames — CSV, track y skeleton guardados.")
    return track_path, skeleton_path, csv_path

# =============================================================================
# PROCESAMIENTO — Video FRONTAL
# =============================================================================
# IDs MediaPipe para ambos lados a la vez (vista de frente)
IDS_FRONTAL = {
    "der": dict(K=26, A=28, H=24, S=12, E=14, W=16, EAR=8, EYE=6, HEEL=30, FOOT=32),
    "izq": dict(K=25, A=27, H=23, S=11, E=13, W=15, EAR=7, EYE=3, HEEL=29, FOOT=31),
}

def procesar_video_frontal(input_path, hombro_cadera_cm, log_fn, progress_fn):
    """
    Procesa un video FRONTAL (de frente a la persona). A diferencia de los
    perfiles, acá se trackean AMBOS lados (der e izq) simultáneamente.
    No se calculan ángulos articulares ni se dibujan arcos: solo el overlay
    estándar de landmarks/skeleton de MediaPipe, más los puntos clave
    (hombros, manos, caderas, orejas) resaltados.
    """
    base_name  = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.dirname(input_path)

    track_path    = os.path.join(output_dir, f"{base_name}_track.mp4")
    skeleton_path = os.path.join(output_dir, f"{base_name}_skeleton.mp4")
    csv_path      = os.path.join(output_dir, f"{base_name}.csv")

    log_fn("[FRONTAL] Iniciando procesamiento…")
    log_fn(f"[FRONTAL] Conversión a cm: {'Sí → ' + str(hombro_cadera_cm) + ' cm' if hombro_cadera_cm else 'No (normalizado)'}")

    ALPHA = 0.7
    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(model_complexity=2, smooth_landmarks=True,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap    = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps_v  = cap.get(cv2.CAP_PROP_FPS) or 30
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    out       = cv2.VideoWriter(track_path,   fourcc, fps_v, (W,H))
    out_black = cv2.VideoWriter(skeleton_path,fourcc, fps_v, (W,H))

    prev_sm    = {}
    scale_cm   = None
    scale_done = False
    data_all   = []
    frame_idx  = 0

    def sm(name, p):
        if name in prev_sm:
            p = [ALPHA*prev_sm[name][0]+(1-ALPHA)*p[0],
                 ALPHA*prev_sm[name][1]+(1-ALPHA)*p[1]]
        prev_sm[name] = p
        return p

    def n2px(p): return (int(p[0]*W), int(p[1]*H))

    pts_colors = {
        "hom": (255,165,0), "mun": (150,150,255),
        "cad": (0,220,255), "ear": (255,255,0),
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        black = np.zeros((H,W,3), dtype=np.uint8)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = pose.process(rgb)

        data = {"frame": frame_idx,
                "hombro_cadera_ref_cm": hombro_cadera_cm if hombro_cadera_cm else np.nan}

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            if hombro_cadera_cm and not scale_done:
                try:
                    ds = []
                    for lado_s, ids in IDS_FRONTAL.items():
                        hx,hy = lm[ids["S"]].x, lm[ids["S"]].y
                        cx,cy = lm[ids["H"]].x, lm[ids["H"]].y
                        d = math.sqrt((hx-cx)**2+(hy-cy)**2)
                        if d > 0.01:
                            ds.append(d)
                    if ds:
                        d_prom    = float(np.mean(ds))
                        scale_cm   = hombro_cadera_cm / d_prom
                        scale_done = True
                        log_fn(f"[FRONTAL] Escala: {scale_cm:.2f} cm/unidad-norm (prom. der/izq)")
                except Exception as ex:
                    log_fn(f"[FRONTAL] Advertencia escala: {ex}")

            # Trackear ambos lados
            puntos_lado = {}
            for lado_s, ids in IDS_FRONTAL.items():
                try:
                    hom = sm(f"hom_{lado_s}", [lm[ids["S"]].x, lm[ids["S"]].y])
                    mun = sm(f"mun_{lado_s}", [lm[ids["W"]].x, lm[ids["W"]].y])
                    cad = sm(f"cad_{lado_s}", [lm[ids["H"]].x, lm[ids["H"]].y])
                    ear = sm(f"ear_{lado_s}", [lm[ids["EAR"]].x, lm[ids["EAR"]].y])
                    rod = sm(f"rod_{lado_s}", [lm[ids["K"]].x, lm[ids["K"]].y])
                    puntos_lado[lado_s] = dict(hom=hom, mun=mun, cad=cad, ear=ear, rod=rod)
                except Exception:
                    puntos_lado[lado_s] = None

            for lado_s, pts in puntos_lado.items():
                if pts is None:
                    continue
                for nombre, p in pts.items():
                    p_px = n2px(p)
                    col = pts_colors.get(nombre, (255,255,255))
                    for img in [frame, black]:
                        draw_joint(img, p_px, col, r=6)

            for i, landmark in enumerate(lm):
                nm = LANDMARK_NAMES[i]
                data[f"{nm}_x_norm"]     = landmark.x
                data[f"{nm}_y_norm"]     = 1.0 - landmark.y
                data[f"{nm}_z_norm"]     = landmark.z
                data[f"{nm}_visibility"] = landmark.visibility
                if scale_cm:
                    data[f"{nm}_x_cm"] = landmark.x * scale_cm
                    data[f"{nm}_y_cm"] = (1.0-landmark.y)*scale_cm
                    data[f"{nm}_z_cm"] = landmark.z * scale_cm

            # angulo_rodilla por lado (para detección de fases, no se dibuja)
            for lado_s, ids in IDS_FRONTAL.items():
                try:
                    rod_p  = [lm[ids["K"]].x, lm[ids["K"]].y]
                    tob_p  = [lm[ids["A"]].x, lm[ids["A"]].y]
                    cad_p  = [lm[ids["H"]].x, lm[ids["H"]].y]
                    ang_r  = calculate_angle(tob_p, rod_p, cad_p)
                    data[f"angulo_rodilla_{lado_s}"] = ang_r
                except Exception:
                    data[f"angulo_rodilla_{lado_s}"] = np.nan

            mp_drawing.draw_landmarks(frame,res.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(200,200,200),thickness=1,circle_radius=2),
                mp_drawing.DrawingSpec(color=(80,80,80),   thickness=1))
            mp_drawing.draw_landmarks(black,res.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(180,180,180),thickness=1,circle_radius=2),
                mp_drawing.DrawingSpec(color=(60,60,60),   thickness=1))

        for li,line in enumerate([f"Frame: {frame_idx}", "Vista: FRONTAL"]):
            for img in [frame, black]:
                cv2.putText(img,line,(12,28+li*22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),3,cv2.LINE_AA)
                cv2.putText(img,line,(12,28+li*22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(220,220,220),1,cv2.LINE_AA)

        data_all.append(data)

        out.write(frame)
        out_black.write(black)
        frame_idx += 1
        progress_fn(min(99, int(frame_idx/total*100)))

    cap.release(); out.release(); out_black.release(); pose.close()

    df = pd.DataFrame(data_all)
    meta   = ["frame","hombro_cadera_ref_cm"]
    angles = [c for c in df.columns if c.startswith("angulo_")]
    lms    = [c for c in df.columns if c not in meta+angles]
    df     = df[meta+angles+lms]
    df.to_csv(csv_path, index=False)

    progress_fn(100)
    log_fn(f"[FRONTAL] ✔ {frame_idx} frames — CSV, track y skeleton guardados.")
    return track_path, skeleton_path, csv_path


# =============================================================================
# ANÁLISIS DE FASES
# =============================================================================

def suavizar(sig):
    sig = np.asarray(sig).astype(float)
    sig = medfilt(sig, kernel_size=5)
    if len(sig) > 11:
        sig = savgol_filter(sig, 11, 3)
    return sig



def detectar_fases(df, lado):
    def _detectar_fases_v4(df, lado):
        """
        Clasifica cada frame en 5 fases del test STS:
            sentado_ini → parando → parado → sentandose → sentado_fin

        Estrategia:
        ──────────
        Usa el ángulo de cadera como señal primaria para los bordes del movimiento
        (es más sensible que la altura de cadera porque cambia antes), y la altura
        de cadera normalizada + ángulo de rodilla para identificar los estados
        estables (sentado / parado).

        Pasos:
        1. Suavizar todas las señales.
        2. Normalizar altura de cadera entre 0-1 (dentro del video).
        3. Calcular velocidad de la altura de cadera (para dirección del movimiento).
        4. Calcular velocidad angular de la cadera (para detectar inicio/fin exacto).
        5. Etiqueta cruda frame a frame con umbrales amplios y seguros.
        6. Limpiar segmentos cortos.
        7. Expandir los bordes de parando/sentandose usando la velocidad angular
           de cadera: el episodio empieza en el primer frame donde la vel supera
           un umbral mínimo, y termina en el último frame antes de que se detenga.
        8. Separar sentado_ini / sentado_fin.
        """
        sufijo    = "der" if lado == "der" else "izq"
        col_cad_y = f"cadera_{sufijo}_y_norm"

        if col_cad_y not in df.columns:
            return ["desconocido"] * len(df), []

        n = len(df)

        # ── Señales suavizadas ──
        rodilla  = suavizar(df["angulo_rodilla"].values)
        cad_y    = suavizar(df[col_cad_y].values)
        ang_cad  = suavizar(df["angulo_cadera"].values) if "angulo_cadera" in df.columns                else np.zeros(n)

        # Velocidad de altura de cadera (para dirección: sube=parando, baja=sentandose)
        vel_alt  = medfilt(np.gradient(cad_y).astype(float), kernel_size=9)

        # Velocidad angular de cadera (para bordes exactos del movimiento)
        vel_ang  = medfilt(np.abs(np.gradient(ang_cad)).astype(float), kernel_size=7)

        # Normalizar altura de cadera 0-1
        y_min       = np.nanmin(cad_y)
        y_max       = np.nanmax(cad_y)
        y_rng       = max(y_max - y_min, 1e-6)
        altura_norm = (cad_y - y_min) / y_rng

        # ── Pasada 1: etiqueta cruda con umbrales amplios ──
        # Sentado:  cadera baja (< 40% del rango) Y rodilla flexionada (< 145°)
        # Parado:   cadera alta (> 65% del rango) Y rodilla extendida  (> 150°)
        # Transición: todo lo demás, diferenciado por dirección de movimiento
        fases = []
        for i in range(n):
            alt = altura_norm[i]
            ang = rodilla[i]
            v   = vel_alt[i]

            if alt < 0.40 and ang < 145:
                fase = "sentado"
            elif alt > 0.65 and ang > 150:
                fase = "parado"
            else:
                # Zona de transición: usar dirección de la cadera
                if v > 0.0001:
                    fase = "parando"
                elif v < -0.0001:
                    fase = "sentandose"
                else:
                    # Sin velocidad clara: heredar del frame anterior
                    fase = fases[i-1] if i > 0 else ("sentado" if alt < 0.5 else "parado")
            fases.append(fase)

        # ── Pasada 2: limpiar segmentos cortos ──
        def build_segs(f):
            segs = []; ini = 0
            for i in range(1, len(f)):
                if f[i] != f[i-1]:
                    segs.append([ini, i-1, f[i-1]]); ini = i
            segs.append([ini, len(f)-1, f[-1]])
            return segs

        for _ in range(5):
            segs    = build_segs(fases)
            changed = False
            for s, (ini, fin, etq) in enumerate(segs):
                if (fin - ini + 1) < 8:
                    prev = segs[s-1][2] if s > 0 else etq
                    for j in range(ini, fin+1):
                        fases[j] = prev
                    changed = True
            if not changed:
                break

        # ── Pasada 3: expandir bordes con velocidad angular de cadera ──
        # Umbral de movimiento: percentil 70 de la vel angular (ignora ruido estático)
        vel_thresh = max(np.percentile(vel_ang, 70), 0.03)

        segs = build_segs(fases)

        # Para cada episodio de "parando": extender inicio hacia atrás y fin hacia adelante
        # mientras la velocidad angular de cadera sea > vel_thresh * 0.25 (umbral suave)
        fases2 = fases.copy()
        for s_idx, (ini, fin, etq) in enumerate(segs):
            if etq not in ("parando", "sentandose"):
                continue

            # Expandir hacia atrás (inicio)
            new_ini = ini
            for i in range(ini - 1, max(ini - 60, -1), -1):
                if fases[i] in ("sentado", "parado"):
                    # Sólo expandir si hay movimiento real
                    if vel_ang[i] > vel_thresh * 0.25:
                        new_ini = i
                    else:
                        break

            # Expandir hacia adelante (fin)
            new_fin = fin
            for i in range(fin + 1, min(fin + 60, n)):
                if fases[i] in ("sentado", "parado"):
                    if vel_ang[i] > vel_thresh * 0.25:
                        new_fin = i
                    else:
                        break

            for j in range(new_ini, new_fin + 1):
                if fases2[j] in ("sentado", "parado", etq):
                    fases2[j] = etq

        fases = fases2

        # Limpiar de nuevo tras la expansión
        for _ in range(3):
            segs    = build_segs(fases)
            changed = False
            for s, (ini, fin, etq) in enumerate(segs):
                if (fin - ini + 1) < 5:
                    prev = segs[s-1][2] if s > 0 else etq
                    for j in range(ini, fin+1):
                        fases[j] = prev
                    changed = True
            if not changed:
                break

        # ── Pasada 4: distinguir sentado_ini / sentado_fin ──
        segs = build_segs(fases)
        idx_first_parando = next(
            (i for i,(a,b,e) in enumerate(segs) if e == "parando"), None)
        idx_last_sentando = next(
            (i for i,(a,b,e) in reversed(list(enumerate(segs))) if e == "sentandose"), None)

        if idx_first_parando is not None:
            limite_ini = segs[idx_first_parando][0]
            for i in range(0, limite_ini):
                if fases[i] == "sentado":
                    fases[i] = "sentado_ini"

        if idx_last_sentando is not None:
            limite_fin = segs[idx_last_sentando][1]
            for i in range(limite_fin + 1, n):
                if fases[i] == "sentado":
                    fases[i] = "sentado_fin"

        segs_final = build_segs(fases)
        return fases, segs_final



    def _detectar_fases_v7(df, lado):
        """
        Clasifica cada frame en 5 fases del test STS:
            sentado_ini → parando → parado → sentandose → sentado_fin

        Estrategia:
        ──────────
        1. Etiqueta cruda: sentado/parado por umbral de altura+rodilla,
           transición por dirección de vel_alt.
        2. Limpieza de segmentos cortos.
        3. Expansión de bordes de parando/sentandose con vel_ang de cadera.
        4. Rellenado explícito del hueco parado→sentandose:
           detecta el último frame donde el "parado" empieza a descender
           (vel_alt < umbral negativo sostenido) y desde ahí hacia adelante
           lo marca como "sentandose" hasta el primer frame estable "sentado".
           Esto resuelve el caso en que la bajada es gradual y la etiqueta
           cruda nunca detecta el inicio del sentandose.
        5. Separar sentado_ini / sentado_fin.
        """
        sufijo    = "der" if lado == "der" else "izq"
        col_cad_y = f"cadera_{sufijo}_y_norm"

        if col_cad_y not in df.columns:
            return ["desconocido"] * len(df), []

        n = len(df)

        # ── Señales suavizadas ──
        rodilla = suavizar(df["angulo_rodilla"].values)
        cad_y   = suavizar(df[col_cad_y].values)
        ang_cad = suavizar(df["angulo_cadera"].values) if "angulo_cadera" in df.columns               else np.zeros(n)

        vel_alt = medfilt(np.gradient(cad_y).astype(float), kernel_size=9)
        vel_ang = medfilt(np.abs(np.gradient(ang_cad)).astype(float), kernel_size=7)

        y_min       = np.nanmin(cad_y)
        y_max       = np.nanmax(cad_y)
        altura_norm = (cad_y - y_min) / max(y_max - y_min, 1e-6)

        # ── Pasada 1: etiqueta cruda ──
        # "sentado seguro": cadera muy baja Y rodilla muy flexionada → siempre sentado,
        # ignorando la velocidad (evita que ruido inicial clasifique mal el sentado_ini).
        # "parado seguro": cadera muy alta Y rodilla muy extendida → siempre parado.
        fases = []
        for i in range(n):
            alt = altura_norm[i]
            ang = rodilla[i]
            v   = vel_alt[i]
            if alt < 0.20 and ang < 130:          # sentado seguro (sin importar vel)
                fase = "sentado"
            elif alt < 0.40 and ang < 145 and abs(v) < 0.002:  # sentado con poca vel
                fase = "sentado"
            elif alt > 0.80 and ang > 155:        # parado seguro (sin importar vel)
                fase = "parado"
            elif alt > 0.65 and ang > 150 and abs(v) < 0.002:  # parado con poca vel
                fase = "parado"
            elif v > 0.0001:
                fase = "parando"
            elif v < -0.0001:
                fase = "sentandose"
            else:
                fase = fases[i-1] if i > 0 else ("sentado" if alt < 0.5 else "parado")
            fases.append(fase)

        # ── Pasada 2: limpiar segmentos cortos ──
        # Los estados estables (sentado, parado) nunca se fusionan entre sí aunque sean
        # cortos; solo se eliminan segmentos cortos de transición (parando/sentandose).
        def build_segs(f):
            segs = []; ini = 0
            for i in range(1, len(f)):
                if f[i] != f[i-1]:
                    segs.append([ini, i-1, f[i-1]]); ini = i
            segs.append([ini, len(f)-1, f[-1]])
            return segs

        for _ in range(5):
            segs    = build_segs(fases)
            changed = False
            for s, (ini, fin, etq) in enumerate(segs):
                dur = fin - ini + 1
                # Solo eliminar segmentos cortos de transición
                if etq in ("parando", "sentandose") and dur < 6:
                    prev = segs[s-1][2] if s > 0 else etq
                    for j in range(ini, fin+1):
                        fases[j] = prev
                    changed = True
                # Segmentos estables muy cortos (< 4f) entre dos iguales también
                elif etq in ("sentado", "parado") and dur < 4:
                    prev = segs[s-1][2] if s > 0 else etq
                    nxt  = segs[s+1][2] if s < len(segs)-1 else etq
                    if prev == nxt:   # solo si los vecinos son iguales
                        for j in range(ini, fin+1):
                            fases[j] = prev
                        changed = True
            if not changed:
                break

        # ── Pasada 3: expandir bordes de parando/sentandose con vel_ang ──
        # IMPORTANTE: la expansión hacia atrás del primer "parando" no puede
        # sobrepasar su inicio original (no debe comerse el sentado_ini).
        vel_thresh = max(np.percentile(vel_ang, 70), 0.03)
        segs       = build_segs(fases)
        fases2     = fases.copy()

        # Ancla: inicio original del primer bloque "parando" (no expandir antes de aquí)
        first_parando_start = next((ini for ini,fin,etq in segs if etq=="parando"), 0)

        for ini, fin, etq in segs:
            if etq not in ("parando", "sentandose"):
                continue
            # Para "parando": no retroceder más allá del inicio original del primer parando
            min_back = first_parando_start if etq == "parando" else max(ini - 60, 0)
            new_ini = ini
            for i in range(ini - 1, max(min_back - 1, -1), -1):
                if fases[i] in ("sentado", "parado") and vel_ang[i] > vel_thresh * 0.20:
                    new_ini = i
                else:
                    break
            new_fin = fin
            for i in range(fin + 1, min(fin + 60, n)):
                if fases[i] in ("sentado", "parado") and vel_ang[i] > vel_thresh * 0.20:
                    new_fin = i
                else:
                    break
            for j in range(new_ini, new_fin + 1):
                if fases2[j] in ("sentado", "parado", etq):
                    fases2[j] = etq

        fases = fases2

        # ── Pasada 4: rellenar hueco sentandose entre el último parado y el sentado ──
        #
        # El problema: cuando la bajada es lenta, los frames al inicio del descenso
        # siguen clasificados como "parado" (alt > 0.65 y rod > 150) aunque la cadera
        # ya esté bajando. La solución es buscar el primer frame DENTRO del bloque
        # "parado" donde vel_alt < -umbral sostenido y desde ahí reclasificar
        # todo como "sentandose" hasta el primer "sentado" que aparezca después.
        #
        # umbral_baja: usamos percentil 15 del vel_alt (valores claramente negativos)
        umbral_baja = min(np.percentile(vel_alt, 20), -0.0005)

        segs = build_segs(fases)

        # Encontrar el último bloque "parado"
        ultimo_parado_seg = None
        for seg in reversed(segs):
            if seg[2] == "parado":
                ultimo_parado_seg = seg
                break

        if ultimo_parado_seg is not None:
            p_ini, p_fin, _ = ultimo_parado_seg

            # Buscar hacia adelante dentro del bloque "parado" el primer frame
            # donde vel_alt < umbral_baja sostenido (al menos 2 frames seguidos)
            inicio_bajada = None
            for i in range(p_ini, p_fin + 1):
                if vel_alt[i] < umbral_baja:
                    # Confirmar con el siguiente frame para evitar ruido
                    if i + 1 < n and vel_alt[i+1] < umbral_baja * 0.5:
                        inicio_bajada = i
                        break

            if inicio_bajada is not None:
                # Reclasificar desde inicio_bajada hasta el primer "sentado" que siga
                primer_sentado_post = n
                for i in range(p_fin + 1, n):
                    if fases[i] == "sentado":
                        primer_sentado_post = i
                        break

                # Extender sentandose también hacia adelante dentro del "sentado"
                # mientras vel_alt siga negativa (cadera todavía bajando)
                fin_sentandose = primer_sentado_post - 1
                for i in range(primer_sentado_post, min(primer_sentado_post + 60, n)):
                    if vel_alt[i] < umbral_baja * 0.3:
                        fin_sentandose = i
                    else:
                        break

                for j in range(inicio_bajada, fin_sentandose + 1):
                    fases[j] = "sentandose"

        # ── Pasada 5: limpiar de nuevo ──
        for _ in range(3):
            segs    = build_segs(fases)
            changed = False
            for s, (ini, fin, etq) in enumerate(segs):
                if (fin - ini + 1) < 5:
                    prev = segs[s-1][2] if s > 0 else etq
                    for j in range(ini, fin+1):
                        fases[j] = prev
                    changed = True
            if not changed:
                break

        # ── Pasada 6: distinguir sentado_ini / sentado_fin ──
        segs = build_segs(fases)
        idx_first_parando = next(
            (i for i,(a,b,e) in enumerate(segs) if e == "parando"), None)
        idx_last_sentando = next(
            (i for i,(a,b,e) in reversed(list(enumerate(segs))) if e == "sentandose"), None)

        if idx_first_parando is not None:
            limite_ini = segs[idx_first_parando][0]
            for i in range(0, limite_ini):
                if fases[i] == "sentado":
                    fases[i] = "sentado_ini"

        if idx_last_sentando is not None:
            limite_fin = segs[idx_last_sentando][1]
            for i in range(limite_fin + 1, n):
                if fases[i] == "sentado":
                    fases[i] = "sentado_fin"

        segs_final = build_segs(fases)
        return fases, segs_final



    fases_v4, segs_v4 = _detectar_fases_v4(df, lado)
    fases_v7, segs_v7 = _detectar_fases_v7(df, lado)

    n = len(df)

    def build_segs(f):
        segs=[]; ini=0
        for i in range(1,len(f)):
            if f[i] != f[i-1]:
                segs.append([ini,i-1,f[i-1]])
                ini=i
        segs.append([ini,len(f)-1,f[-1]])
        return segs

    # Inicio del sentandose según v7
    idx_inicio_sentandose_v7 = next((ini for ini,fin,etq in segs_v7 if etq=="sentandose"), None)

    # Inicio del primer parando según v4
    idx_inicio_parando_v4 = next((ini for ini,fin,etq in segs_v4 if etq=="parando"), None)

    fases = ["desconocido"] * n

    # sentado_ini + parando exactamente como v4
    if idx_inicio_parando_v4 is not None:
        for i in range(idx_inicio_parando_v4):
            fases[i] = "sentado_ini"
        for i,f in enumerate(fases_v4):
            if f == "parando":
                fases[i] = "parando"

    # sentandose + sentado_fin exactamente como v7
    if idx_inicio_sentandose_v7 is not None:
        for i in range(idx_inicio_sentandose_v7, n):
            if fases_v7[i] == "sentandose":
                fases[i] = "sentandose"
            elif fases_v7[i] == "sentado_fin":
                fases[i] = "sentado_fin"

    # parado híbrido entre ambos límites
    fin_parando_v4 = max((i for i,f in enumerate(fases_v4) if f=="parando"), default=None)

    if fin_parando_v4 is not None and idx_inicio_sentandose_v7 is not None:
        for i in range(fin_parando_v4+1, idx_inicio_sentandose_v7):
            fases[i] = "parado"

    # rellenar posibles huecos residuales con clasificación v7
    for i in range(n):
        if fases[i] == "desconocido":
            fases[i] = fases_v7[i]

    # ── Enforce orden biológico ──
    # El STS tiene un único orden posible: sentado_ini → parando → parado → sentandose → sentado_fin.
    # Nunca puede aparecer "sentandose" antes del primer "parado", ni "parando" después
    # del último "parado". Corregimos frames fuera de orden reemplazándolos por el
    # estado esperado según el contexto (parando si estamos antes del parado, etc.)
    orden = ["sentado_ini", "parando", "parado", "sentandose", "sentado_fin"]

    def fase_idx(f):
        try: return orden.index(f)
        except ValueError: return -1

    # Encontrar el frame donde comienza y termina "parado"
    frames_parado = [i for i,f in enumerate(fases) if f == "parado"]
    if frames_parado:
        primer_parado = frames_parado[0]
        ultimo_parado = frames_parado[-1]

        # Antes del primer "parado": solo puede haber sentado_ini o parando
        for i in range(primer_parado):
            if fases[i] in ("sentandose", "sentado_fin", "parado"):
                fases[i] = "parando" if i >= (idx_inicio_parando_v4 or 0) else "sentado_ini"

        # Después del último "parado": solo puede haber sentandose o sentado_fin
        for i in range(ultimo_parado + 1, n):
            if fases[i] in ("sentado_ini", "parando", "parado"):
                fases[i] = "sentandose" if fases_v7[i] != "sentado_fin" else "sentado_fin"

    # Limpieza final: eliminar segmentos muy cortos (< 5 frames) que puedan haber quedado
    for _ in range(3):
        segs    = build_segs(fases)
        changed = False
        for s, (ini, fin, etq) in enumerate(segs):
            if (fin - ini + 1) < 5:
                vecino = segs[s-1][2] if s > 0 else (segs[s+1][2] if s+1 < len(segs) else etq)
                for j in range(ini, fin+1):
                    fases[j] = vecino
                changed = True
        if not changed:
            break

    return fases, build_segs(fases)

def calcular_vel_angular_brazo(df_seg, fps=30):
    if "angulo_brazo" not in df_seg.columns:
        return np.nan, np.nan
    ang = df_seg["angulo_brazo"].values.astype(float)
    vel = np.abs(np.diff(ang)) * fps
    if len(vel) == 0:
        return np.nan, np.nan
    return float(np.nanmedian(vel)), float(np.nanmax(vel))


def calcular_vel_angular_cadera(df_seg, fps=30):
    if "angulo_cadera" not in df_seg.columns:
        return np.nan, np.nan
    ang = df_seg["angulo_cadera"].values.astype(float)
    vel = np.abs(np.diff(ang)) * fps
    if len(vel) == 0:
        return np.nan, np.nan
    return float(np.nanmedian(vel)), float(np.nanmax(vel))


def analizar_fase(df_seg, lado, tipo_fase="transicion"):
    """
    Calcula métricas según el tipo de fase:
      - sentado_ini / sentado_fin : solo ángulos iniciales (primer frame)
      - parando / sentandose      : mínimos, máximos, ROM, velocidades angulares
      - parado                    : coeficiente de estabilidad DS hombro-X
    """
    if len(df_seg) < 3:
        return None

    sufijo = "der" if lado == "der" else "izq"
    hom    = f"hombro_{sufijo}"
    fps    = 30
    res    = {"n_frames": len(df_seg), "duracion_s": len(df_seg)/fps}

    angulos = ["tobillo","rodilla","cadera","brazo","cuello_vertical","cabeza_cuello"]

    # Valores inicio y fin siempre (útil para todas las fases)
    for ang in angulos:
        col = f"angulo_{ang}"
        res[f"{ang}_ini"] = float(df_seg[col].iloc[0])  if col in df_seg.columns else np.nan
        res[f"{ang}_fin"] = float(df_seg[col].iloc[-1]) if col in df_seg.columns else np.nan

    if tipo_fase in ("parando", "sentandose"):
        # Mínimos: cuello, cabeza, rodilla, cadera, tobillo
        for ang in ["cuello_vertical","cabeza_cuello","rodilla","cadera","tobillo"]:
            col = f"angulo_{ang}"
            res[f"{ang}_min"] = float(df_seg[col].min()) if col in df_seg.columns else np.nan

        # Máximos: cuello, cabeza, brazo
        for ang in ["cuello_vertical","cabeza_cuello","brazo"]:
            col = f"angulo_{ang}"
            res[f"{ang}_max"] = float(df_seg[col].max()) if col in df_seg.columns else np.nan

        # ROM
        for ang in angulos:
            col = f"angulo_{ang}"
            if col in df_seg.columns:
                vals = df_seg[col].values.astype(float)
                res[f"{ang}_rom"] = float(np.nanmax(vals) - np.nanmin(vals))
            else:
                res[f"{ang}_rom"] = np.nan

        # Velocidad angular brazo y cadera
        med_b, mx_b = calcular_vel_angular_brazo(df_seg, fps)
        res["vel_ang_brazo_mediana"] = med_b
        res["vel_ang_brazo_max"]     = mx_b

        med_c, mx_c = calcular_vel_angular_cadera(df_seg, fps)
        res["vel_ang_cadera_mediana"] = med_c
        res["vel_ang_cadera_max"]     = mx_c

    if tipo_fase == "parado":
        try:
            # Usar coordenadas en cm si están disponibles, sino normalizado * 100
            col_cm   = f"{hom}_x_cm"
            col_norm = f"{hom}_x_norm"
            if col_cm in df_seg.columns:
                hx = df_seg[col_cm].values.astype(float)
                res["estabilidad_ds"]      = float(np.std(hx))
                res["estabilidad_ds_unit"] = "cm"
            elif col_norm in df_seg.columns:
                hx = df_seg[col_norm].values.astype(float)
                res["estabilidad_ds"]      = float(np.std(hx))
                res["estabilidad_ds_unit"] = "norm"
            else:
                res["estabilidad_ds"]      = np.nan
                res["estabilidad_ds_unit"] = ""
        except:
            res["estabilidad_ds"]      = np.nan
            res["estabilidad_ds_unit"] = ""

    return res


def analizar_csv(csv_path, lado):
    df  = pd.read_csv(csv_path)
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].interpolate(limit_direction="both")

    # El _der usa el mismo code path que el _izq.
    # Copiamos cadera_der_y_norm → cadera_izq_y_norm.
    # Además, si la señal de cadera del _der está invertida respecto al _izq
    # (en algunos setups el DER graba con la persona mirando al otro lado y la Y
    # baja cuando la persona se para en vez de subir), la invertimos para que
    # vel_alt > 0 siempre signifique "parando" y vel_alt < 0 "sentandose".
    if lado == "der":
        df_fases = df.copy()
        col_src  = "cadera_der_y_norm"
        if col_src in df_fases.columns:
            señal = df_fases[col_src].values.astype(float)
            # Detectar si la señal está invertida: comparamos la media del
            # primer 20% de frames vs el medio del video.
            # Si la persona empieza sentada, la cadera debería estar más baja al inicio.
            # Si el primer cuarto es MAYOR que el centro, la Y está invertida.
            n_v = len(señal)
            q1  = np.nanmean(señal[:max(1, n_v//5)])
            mid = np.nanmean(señal[n_v//3: 2*n_v//3])
            if q1 > mid:
                # La señal baja cuando debería subir → invertir respecto al centro
                señal = 1.0 - señal
            df_fases["cadera_izq_y_norm"] = señal
        fases, segs = detectar_fases(df_fases, "izq")
    else:
        fases, segs = detectar_fases(df, lado)

    df["fase"] = fases

    df["fase"] = fases

    df["fase"] = fases

    resultado = {
        "df": df, "fases": fases, "segmentos": segs,
        "sentado_ini":  [],
        "parando":      [],
        "parado":       [],
        "sentandose":   [],
        "sentado_fin":  [],
    }

    for ini, fin, etiqueta in segs:
        if etiqueta not in resultado:
            continue
        seg = df.iloc[ini:fin+1]
        m   = analizar_fase(seg, lado, tipo_fase=etiqueta)
        if m is None:
            continue
        m["frame_ini"] = ini
        m["frame_fin"] = fin
        resultado[etiqueta].append(m)

    return resultado


# =============================================================================
# ANÁLISIS — Video FRONTAL (fases promediadas + métricas de simetría)
# =============================================================================

def detectar_fases_frontal(df):
    """
    Corre la detección de fases estándar (detectar_fases) de forma
    independiente sobre el lado derecho y el lado izquierdo del video
    frontal, y promedia los índices de transición de cada segmento para
    obtener una única secuencia de fases "consensuada".

    Para poder reutilizar detectar_fases() (que espera columnas genéricas
    'angulo_cadera', 'angulo_rodilla' y 'cadera_<lado>_y_norm'), se arma
    un df temporal por lado con esas columnas renombradas a partir de
    las columnas frontales (que ya tienen sufijo _der/_izq).
    """
    n = len(df)

    fases_por_lado = {}
    segs_por_lado  = {}
    for lado_s in ("der", "izq"):
        col_cad_y = f"cadera_{lado_s}_y_norm"
        col_rod   = f"angulo_rodilla_{lado_s}"
        if col_cad_y not in df.columns or col_rod not in df.columns:
            continue

        # angulo_cadera no existe en frontal (no se calculan ángulos de tronco);
        # se sustituye por una señal proporcional a la altura de cadera invertida,
        # que detectar_fases() solo usa para afinar bordes (vel_ang), por lo que
        # basta con que sea una señal suave y monótonamente relacionada al movimiento.
        df_tmp = pd.DataFrame({
            f"cadera_{lado_s}_y_norm": df[col_cad_y].values,
            "angulo_rodilla":          df[col_rod].values,
            "angulo_cadera":           df[col_rod].values,
        })
        fases_l, segs_l = detectar_fases(df_tmp, lado_s)
        fases_por_lado[lado_s] = fases_l
        segs_por_lado[lado_s]  = segs_l

    if not fases_por_lado:
        return ["desconocido"] * n, []

    if len(fases_por_lado) == 1:
        unico = list(fases_por_lado.values())[0]
        def build_segs(f):
            segs = []; ini = 0
            for i in range(1, len(f)):
                if f[i] != f[i-1]:
                    segs.append([ini, i-1, f[i-1]]); ini = i
            segs.append([ini, len(f)-1, f[-1]])
            return segs
        return unico, build_segs(unico)

    # ── Promediar índices de transición entre las 5 fases canónicas ──
    orden_fases = ["sentado_ini", "parando", "parado", "sentandose", "sentado_fin"]

    def primer_frame(fases_l, etiqueta):
        for i, f in enumerate(fases_l):
            if f == etiqueta:
                return i
        return None

    def ultimo_frame(fases_l, etiqueta):
        for i in range(len(fases_l)-1, -1, -1):
            if fases_l[i] == etiqueta:
                return i
        return None

    f_der = fases_por_lado.get("der")
    f_izq = fases_por_lado.get("izq")

    # Transiciones clave: fin de sentado_ini / ini de parando, fin de parando / ini de parado,
    # fin de parado / ini de sentandose, fin de sentandose / ini de sentado_fin.
    def transicion_promedio(et_a, et_b):
        """Promedia el punto medio de la transición et_a → et_b entre der e izq."""
        candidatos = []
        for fl in (f_der, f_izq):
            if fl is None:
                continue
            fin_a = ultimo_frame(fl, et_a)
            ini_b = primer_frame(fl, et_b)
            if fin_a is not None and ini_b is not None and ini_b > fin_a:
                candidatos.append((fin_a + ini_b) / 2.0)
            elif ini_b is not None:
                candidatos.append(float(ini_b))
            elif fin_a is not None:
                candidatos.append(float(fin_a) + 1.0)
        if not candidatos:
            return None
        return int(round(np.mean(candidatos)))

    t1 = transicion_promedio("sentado_ini", "parando")
    t2 = transicion_promedio("parando", "parado")
    t3 = transicion_promedio("parado", "sentandose")
    t4 = transicion_promedio("sentandose", "sentado_fin")

    cortes = [c for c in [t1, t2, t3, t4] if c is not None]
    cortes = sorted(set(c for c in cortes if 0 <= c <= n))

    # Construir secuencia de fases a partir de los cortes válidos disponibles
    fases = ["desconocido"] * n
    limites = [0] + cortes + [n]
    etiquetas_disponibles = []
    # Determinar qué etiquetas corresponden a cada tramo según qué cortes existen
    nombres_cortes = []
    if t1 is not None: nombres_cortes.append(("sentado_ini","parando"))
    if t2 is not None: nombres_cortes.append(("parando","parado"))
    if t3 is not None: nombres_cortes.append(("parado","sentandose"))
    if t4 is not None: nombres_cortes.append(("sentandose","sentado_fin"))

    # Mapear: antes del primer corte = primera etiqueta de la primera transición disponible
    if nombres_cortes:
        secuencia_etq = [nombres_cortes[0][0]] + [b for (_,b) in nombres_cortes]
        for i in range(len(limites)-1):
            ini_l, fin_l = limites[i], limites[i+1]
            etq = secuencia_etq[i] if i < len(secuencia_etq) else secuencia_etq[-1]
            for j in range(ini_l, fin_l):
                fases[j] = etq
    else:
        # Sin transiciones detectables: usar la del lado con más info
        fases = f_der if f_der is not None else f_izq

    def build_segs(f):
        segs = []; ini = 0
        for i in range(1, len(f)):
            if f[i] != f[i-1]:
                segs.append([ini, i-1, f[i-1]]); ini = i
        segs.append([ini, len(f)-1, f[-1]])
        return segs

    return fases, build_segs(fases)


def _diff_y_metricas(df_seg, nombre_par):
    """
    Para un par de columnas Y (der/izq) ya normalizadas a 0-1 (1=arriba),
    devuelve diferencia media y máxima (valor absoluto) en ese segmento.
    nombre_par: ej. "hombro" → busca hombro_der_y_norm / hombro_izq_y_norm
    """
    col_der = f"{nombre_par}_der_y_norm"
    col_izq = f"{nombre_par}_izq_y_norm"
    if col_der not in df_seg.columns or col_izq not in df_seg.columns:
        return np.nan, np.nan
    diff = np.abs(df_seg[col_der].values.astype(float) - df_seg[col_izq].values.astype(float))
    if len(diff) == 0:
        return np.nan, np.nan
    return float(np.nanmean(diff)), float(np.nanmax(diff))


def _dist_x_manos(df_seg, usar_cm):
    """Distancia en X entre muñeca derecha e izquierda (cm si hay escala, sino norm)."""
    suf = "_x_cm" if usar_cm else "_x_norm"
    col_der = f"muneca_der{suf}"
    col_izq = f"muneca_izq{suf}"
    if col_der not in df_seg.columns or col_izq not in df_seg.columns:
        return None
    return np.abs(df_seg[col_der].values.astype(float) - df_seg[col_izq].values.astype(float))


def analizar_fase_frontal(df_seg, tipo_fase, usar_cm):
    """
    Calcula, para un segmento de fase del video frontal:
      - Diferencia media y máxima en Y de: hombro, mano (muñeca), cadera, oreja
      - Distancia en X entre manos:
          sentado_ini / sentado_fin → valor inicial / final (un solo número)
          sentandose / parando / parado → media y máxima
    """
    if len(df_seg) < 1:
        return None

    res = {"n_frames": len(df_seg), "duracion_s": len(df_seg)/30.0}

    pares = [
        ("hombro", "hombro"), ("mano", "muneca"),
        ("cadera", "cadera"), ("oreja", "oreja"),
    ]
    for etiqueta, nombre_col in pares:
        media, maxi = _diff_y_metricas(df_seg, nombre_col)
        res[f"diff_y_{etiqueta}_media"] = media
        res[f"diff_y_{etiqueta}_max"]   = maxi

    dist_x = _dist_x_manos(df_seg, usar_cm)
    res["dist_x_unit"] = "cm" if usar_cm else "norm"

    if tipo_fase == "sentado_ini":
        res["dist_x_manos_inicial"] = float(dist_x[0]) if dist_x is not None and len(dist_x) else np.nan
    elif tipo_fase == "sentado_fin":
        res["dist_x_manos_final"] = float(dist_x[-1]) if dist_x is not None and len(dist_x) else np.nan
    else:  # parando, parado, sentandose
        if dist_x is not None and len(dist_x):
            res["dist_x_manos_media"] = float(np.nanmean(dist_x))
            res["dist_x_manos_max"]   = float(np.nanmax(dist_x))
        else:
            res["dist_x_manos_media"] = np.nan
            res["dist_x_manos_max"]   = np.nan

    return res


def analizar_csv_frontal(csv_path, segs_der=None):
    """
    Las fases del frontal se toman SIEMPRE de los segmentos del perfil derecho
    (segs_der). No hay detección propia: los mismos frames de inicio/fin de cada
    fase que el _der se aplican directamente al CSV frontal.
    """
    df  = pd.read_csv(csv_path)
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].interpolate(limit_direction="both")

    if not segs_der:
        raise ValueError(
            "El frontal requiere que el perfil izquierdo esté procesado primero. "
            "Procesá el video _izq antes o junto con el frontal.")

    n     = len(df)
    fases = ["desconocido"] * n
    for ini, fin, etq in segs_der:
        for i in range(max(0, ini), min(n, fin + 1)):
            fases[i] = etq
    segs = segs_der

    df["fase"] = fases

    usar_cm = "muneca_der_x_cm" in df.columns and "muneca_izq_x_cm" in df.columns

    resultado = {
        "df": df, "fases": fases, "segmentos": segs,
        "sentado_ini":  [],
        "parando":      [],
        "parado":       [],
        "sentandose":   [],
        "sentado_fin":  [],
    }

    for ini, fin, etiqueta in segs:
        if etiqueta not in resultado:
            continue
        seg = df.iloc[ini:fin+1]
        m   = analizar_fase_frontal(seg, tipo_fase=etiqueta, usar_cm=usar_cm)
        if m is None:
            continue
        m["frame_ini"] = ini
        m["frame_fin"] = fin
        resultado[etiqueta].append(m)

    return resultado


# =============================================================================
# HELPERS UI
# =============================================================================

def mk_lbl(parent, text, size=10, bold=False, color=C_TEXT, **kw):
    return tk.Label(parent, text=text,
                    font=("Segoe UI",size,"bold" if bold else "normal"),
                    fg=color, bg=parent.cget("bg"), **kw)

def mk_card(parent, **kw):
    return tk.Frame(parent, bg=C_CARD, bd=0, **kw)

def metric_row(parent, label, value, unit="", color=C_TEXT):
    row = tk.Frame(parent, bg=parent.cget("bg"))
    row.pack(fill="x", padx=6, pady=2)
    tk.Label(row, text=label, font=("Segoe UI",10), fg=C_SUB,
             bg=row.cget("bg"), anchor="w").pack(side="left")
    v = f"{value:.2f}" if isinstance(value,float) and not np.isnan(value) else \
        (str(value) if not (isinstance(value,float) and np.isnan(value)) else "—")
    tk.Label(row, text=f"{v} {unit}".strip(), font=("Segoe UI",10),
             fg=color, bg=row.cget("bg"), anchor="e").pack(side="right")

def mk_entry(parent, textvariable, width=12):
    return tk.Entry(parent, textvariable=textvariable, width=width,
                    bg=C_CARD, fg=C_TEXT, insertbackground=C_TEXT,
                    font=("Segoe UI",10), bd=0,
                    highlightthickness=1, highlightcolor=C_TEXT)

def mk_browse_btn(parent, command, text="Examinar…"):
    return tk.Button(parent, text=text, command=command,
                     bg=C_CARD, fg=C_TEXT, bd=0, padx=10, pady=4,
                     font=("Segoe UI",10), cursor="hand2")

# =============================================================================
# GENERACIÓN DE PDF
# =============================================================================

def generar_pdf(resultado, lado, chart_fig, save_path):
    """
    Genera un PDF con los datos del panel derecho (métricas por fase) y el gráfico.
    """
    doc = SimpleDocTemplate(
        save_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()
    title_style  = ParagraphStyle("titulo",  parent=styles["Title"],
                                   fontSize=16, spaceAfter=6,
                                   textColor=colors.HexColor(C_TEXT))
    h1_style     = ParagraphStyle("h1",      parent=styles["Heading1"],
                                   fontSize=13, spaceBefore=14, spaceAfter=4,
                                   textColor=colors.HexColor(C_TEXT))
    h2_style     = ParagraphStyle("h2",      parent=styles["Heading2"],
                                   fontSize=11, spaceBefore=8, spaceAfter=2,
                                   textColor=colors.HexColor(C_TEXT))
    body_style   = ParagraphStyle("body",    parent=styles["Normal"],
                                   fontSize=9,  leading=13)

    story = []

    # Título
    titulo_lado = "Frontal" if lado == "frontal" else f"Lado {lado.upper()}"
    story.append(Paragraph(f"Informe Biomecánico STS — {titulo_lado}", title_style))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor(C_TEXT), spaceAfter=8))

    # ── Gráfico ──
    story.append(Paragraph("Gráfico de ángulos articulares" if lado != "frontal"
                            else "Gráfico de alturas Y (der/izq)", h1_style))
    buf = io.BytesIO()
    chart_fig.savefig(buf, format="png", dpi=120,
                      bbox_inches="tight", facecolor=C_BG)
    buf.seek(0)
    story.append(RLImage(buf, width=16*cm, height=7*cm))
    story.append(Spacer(1, 10))

    # Colores de leyenda en el PDF
    col_leyenda = {
        "parado":      colors.HexColor(C_GREEN),
        "parando":     colors.HexColor(C_YELLOW),
        "sentandose":  colors.HexColor(C_RED),
        "sentado_ini": colors.HexColor(C_ACCENT2),
        "sentado_fin": colors.HexColor(C_ACCENT2),
    }

    # ── Secciones en orden ──
    ORDEN = [
        ("sentado_ini",  "Sentado inicial"),
        ("parando",      "Parándose"),
        ("parado",       "Parado"),
        ("sentandose",   "Sentándose"),
        ("sentado_fin",  "Sentado final"),
    ]

    def fmt(v, unit="°"):
        if isinstance(v, float) and np.isnan(v):
            return "—"
        if isinstance(v, float):
            return f"{v:.2f} {unit}".strip()
        return str(v)

    def tabla_datos(filas):
        tdata = []
        filas_encabezado = []

        for i, (l, v) in enumerate(filas):

            if l.startswith("•"):
                filas_encabezado.append(i)

                tdata.append([
                    Paragraph(f"<b>{l}</b>", body_style),
                    Paragraph("", body_style)
                ])
            else:
                tdata.append([
                    Paragraph(l, body_style),
                    Paragraph(v, body_style)
                ])

        t = Table(tdata, colWidths=[9*cm, 7*cm])

        estilos = [
            ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor(C_TEXT)),
            ("ROWBACKGROUNDS", (0,0), (-1,-1),
                [colors.HexColor(C_BG), colors.HexColor(C_CARD)]),
            ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor(C_BG)),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("TOPPADDING", (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
        ]

        # Encabezados especiales
        for fila in filas_encabezado:
            estilos.extend([
                ("BACKGROUND", (0,fila), (-1,fila),
                 colors.HexColor(C_PANEL)),
                ("SPAN", (0,fila), (1,fila)),
                ("FONTNAME", (0,fila), (-1,fila),
                 "Helvetica-Bold"),
            ])

        t.setStyle(TableStyle(estilos))

        return t
    for clave, titulo in ORDEN:
        items = resultado.get(clave, [])
        if not items:
            continue

        story.append(Paragraph(titulo, h1_style))

        for ep_i, m in enumerate(items):
            ep_label = f"Episodio {ep_i+1}  (frames {m['frame_ini']}→{m['frame_fin']}  |  {m['duracion_s']:.1f} s)"
            story.append(Paragraph(ep_label, h2_style))

            filas = []

            if lado == "frontal":
                unit_x = m.get("dist_x_unit", "")
                filas.append(("• Diferencia Y media (der-izq) •", ""))
                for campo, label in [
                    ("hombro","Hombro"), ("mano","Mano"),
                    ("cadera","Cadera"), ("oreja","Oreja"),
                ]:
                    filas.append((label, fmt(m.get(f"diff_y_{campo}_media", np.nan), "")))

                filas.append(("• Diferencia Y máxima (der-izq) •", ""))
                for campo, label in [
                    ("hombro","Hombro"), ("mano","Mano"),
                    ("cadera","Cadera"), ("oreja","Oreja"),
                ]:
                    filas.append((label, fmt(m.get(f"diff_y_{campo}_max", np.nan), "")))

                filas.append(("• Distancia X entre manos •", ""))
                if clave == "sentado_ini":
                    filas.append(("Inicial", fmt(m.get("dist_x_manos_inicial", np.nan), unit_x)))
                elif clave == "sentado_fin":
                    filas.append(("Final", fmt(m.get("dist_x_manos_final", np.nan), unit_x)))
                else:
                    filas.append(("Media",  fmt(m.get("dist_x_manos_media", np.nan), unit_x)))
                    filas.append(("Máxima", fmt(m.get("dist_x_manos_max",   np.nan), unit_x)))

            elif clave in ("sentado_ini", "sentado_fin"):
                # Solo ángulos iniciales
                for ang, label in [
                    ("tobillo","Tobillo"), ("rodilla","Rodilla"), ("cadera","Cadera"),
                    ("brazo","Brazo"), ("cuello_vertical","Cuello-vertical"),
                    ("cabeza_cuello","Cabeza-cuello"),
                ]:
                    filas.append((f"{label} inicial", fmt(m.get(f"{ang}_ini", np.nan))))

            elif clave in ("parando", "sentandose"):
                # Ángulos ini/fin
                filas.append(("• Inicio / Fin •", ""))
                for ang, label in [
                    ("tobillo","Tobillo"), ("rodilla","Rodilla"), ("cadera","Cadera"),
                    ("brazo","Brazo"), ("cuello_vertical","Cuello-vertical"),
                    ("cabeza_cuello","Cabeza-cuello"),
                ]:
                    filas.append((f"{label} inicio", fmt(m.get(f"{ang}_ini", np.nan))))
                    filas.append((f"{label} fin",    fmt(m.get(f"{ang}_fin", np.nan))))

                # Mínimos
                filas.append(("• Mínimos •", ""))
                for ang, label in [
                    ("cuello_vertical","Cuello-vertical"), ("cabeza_cuello","Cabeza-cuello"),
                    ("rodilla","Rodilla"), ("cadera","Cadera"), ("tobillo","Tobillo"),
                ]:
                    filas.append((f"{label} mín", fmt(m.get(f"{ang}_min", np.nan))))

                # Máximos
                filas.append(("• Máximos •", ""))
                for ang, label in [
                    ("cuello_vertical","Cuello-vertical"), ("cabeza_cuello","Cabeza-cuello"),
                    ("brazo","Brazo"),
                ]:
                    filas.append((f"{label} máx", fmt(m.get(f"{ang}_max", np.nan))))

                # ROM
                filas.append(("• ROM •", ""))
                for ang, label in [
                    ("tobillo","Tobillo"), ("rodilla","Rodilla"), ("cadera","Cadera"),
                    ("brazo","Brazo"), ("cuello_vertical","Cuello-vertical"),
                    ("cabeza_cuello","Cabeza-cuello"),
                ]:
                    filas.append((f"{label} ROM", fmt(m.get(f"{ang}_rom", np.nan))))

                # Velocidades
                filas.append(("• Velocidad angular brazo •", ""))
                filas.append(("Mediana", fmt(m.get("vel_ang_brazo_mediana", np.nan), "°/s")))
                filas.append(("Máxima",  fmt(m.get("vel_ang_brazo_max",     np.nan), "°/s")))
                filas.append(("• Velocidad angular cadera •", ""))
                filas.append(("Mediana", fmt(m.get("vel_ang_cadera_mediana", np.nan), "°/s")))
                filas.append(("Máxima",  fmt(m.get("vel_ang_cadera_max",     np.nan), "°/s")))

            elif clave == "parado":
                filas.append(("Estabilidad DS hombro-X",
                               fmt(m.get("estabilidad_ds", np.nan), "")))

            if filas:
                story.append(tabla_datos(filas))
            story.append(Spacer(1, 6))

    doc.build(story)


# =============================================================================
# APLICACIÓN
# =============================================================================

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Análisis Biomecánico STS")
        self.state("zoomed")
        self.configure(bg=C_BG)

        self._results = {
            "der":     {"track":None,"skeleton":None,"csv":None,"resultado":None},
            "izq":     {"track":None,"skeleton":None,"csv":None,"resultado":None},
            "frontal": {"track":None,"skeleton":None,"csv":None,"resultado":None},
        }

        self._cap           = None
        self._playing       = False
        self._fps_v         = 30
        self._current_frame = 0
        self._total_frames  = 0
        self._updating_sl   = False

        self._lado_vis    = tk.StringVar(value="der")
        self._vidtype_vis = tk.StringVar(value="track")

        self._proc_running  = False
        self._chart_cursor  = None

        # Widgets de secciones desplegables (se crean en _build_vis_tab)
        self._section_frames  = {}   # clave → Frame contenido
        self._section_btns    = {}   # clave → botón toggle
        self._section_visible = {}   # clave → bool

        self._build_ui()

    # ─────────────────────────────────────────────
    # UI PRINCIPAL
    # ─────────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",      background=C_BG, borderwidth=0)
        style.configure("TNotebook.Tab",  background=C_CARD, foreground=C_TEXT,
                         padding=[14,6],  font=("Segoe UI",11,"bold"))
        style.map("TNotebook.Tab",
                  background=[("selected",C_PANEL)],
                  foreground=[("selected",C_TEXT)])

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        nb.bind("<<NotebookTabChanged>>", self._on_tab_change)
        self._nb = nb

        self._tab_proc = tk.Frame(nb, bg=C_BG)
        nb.add(self._tab_proc, text="  ⚙  Procesar  ")
        self._build_proc_tab()

        self._tab_vis = tk.Frame(nb, bg=C_BG)
        nb.add(self._tab_vis, text="  📊  Visualizar  ")
        self._build_vis_tab()

    def _on_tab_change(self, event):
        idx = self._nb.index(self._nb.select())
        if idx == 1:
            self._refresh_vis()

    # ─────────────────────────────────────────────
    # PESTAÑA PROCESAR  (sin cambios)
    # ─────────────────────────────────────────────

    def _build_proc_tab(self):
        p = self._tab_proc

        hdr = tk.Frame(p, bg=C_PANEL, pady=10)
        hdr.pack(fill="x", padx=0)
        mk_lbl(hdr,"  ⚙  Configuración del procesamiento",14,bold=True,color=C_TEXT).pack(side="left",padx=10)

        cfg = tk.Frame(p, bg=C_PANEL, padx=20, pady=14)
        cfg.pack(fill="x", padx=0)

        row_der = tk.Frame(cfg, bg=C_PANEL); row_der.pack(fill="x", pady=5)
        mk_lbl(row_der,"Video lateral derecho:", 10, bold=True, color=C_TEXT).pack(side="left", padx=(0,8))
        self._path_der = tk.StringVar()
        mk_entry(row_der, self._path_der, width=55).pack(side="left", ipady=4)
        mk_browse_btn(row_der, lambda: self._browse("der")).pack(side="left", padx=8)

        row_izq = tk.Frame(cfg, bg=C_PANEL); row_izq.pack(fill="x", pady=5)
        mk_lbl(row_izq,"Video lateral izquierdo:", 10, bold=True, color=C_TEXT).pack(side="left", padx=(0,8))
        self._path_izq = tk.StringVar()
        mk_entry(row_izq, self._path_izq, width=55).pack(side="left", ipady=4)
        mk_browse_btn(row_izq, lambda: self._browse("izq")).pack(side="left", padx=8)

        row_front = tk.Frame(cfg, bg=C_PANEL); row_front.pack(fill="x", pady=5)
        mk_lbl(row_front,"Video frontal:", 10, bold=True, color=C_TEXT).pack(side="left", padx=(0,8))
        self._path_frontal = tk.StringVar()
        mk_entry(row_front, self._path_frontal, width=55).pack(side="left", ipady=4)
        mk_browse_btn(row_front, lambda: self._browse("frontal")).pack(side="left", padx=8)

        row_cm = tk.Frame(cfg, bg=C_PANEL); row_cm.pack(fill="x", pady=(12,2))
        mk_lbl(row_cm,"Distancia hombro → cadera (cm):", 10, color=C_SUB).pack(side="left", padx=(0,8))
        self._cm_var = tk.StringVar()
        mk_entry(row_cm, self._cm_var, width=10).pack(side="left", ipady=4)
        mk_lbl(row_cm,"  (vacío = coordenadas normalizadas 0-1)", 9, color=C_SUB).pack(side="left")

        btn_frame = tk.Frame(p, bg=C_BG)
        btn_frame.pack(pady=10)
        self._btn_proc = tk.Button(btn_frame, text="▶  Procesar videos",
                                    command=self._start_processing,
                                    bg=C_PANEL, fg=C_TEXT, bd=0, padx=24, pady=10,
                                    font=("Segoe UI",12,"bold"), cursor="hand2",
                                    activebackground=C_PANEL)
        self._btn_proc.pack(side="left", padx=8)

        prog_frame = tk.Frame(p, bg=C_BG)
        prog_frame.pack(fill="x", padx=20, pady=2)
        self._status_lbl = mk_lbl(prog_frame,"Listo.",10,color=C_SUB)
        self._status_lbl.pack(anchor="w")
        self._progress_var = tk.DoubleVar(value=0)
        self._pbar = ttk.Progressbar(prog_frame, variable=self._progress_var,
                                      maximum=100, length=700, mode="determinate")
        self._pbar.pack(fill="x", pady=4)

        log_frame = tk.Frame(p, bg=C_BG)
        log_frame.pack(fill="both", expand=True, padx=20, pady=4)
        mk_lbl(log_frame,"Procesamiento:",9,color=C_SUB).pack(anchor="w")
        scroll_f = tk.Frame(log_frame, bg=C_BG)
        scroll_f.pack(fill="both", expand=True)
        self._log_text = tk.Text(scroll_f, bg=C_CARD, fg=C_TEXT,
                                  font=("Consolas",10), wrap="word", bd=0,
                                  highlightthickness=0)
        sb = ttk.Scrollbar(scroll_f, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=sb.set)
        self._log_text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _browse(self, lado):
        nombres = {"der":"derecho", "izq":"izquierdo", "frontal":"frontal"}
        path = filedialog.askopenfilename(
            title=f"Seleccionar video {nombres.get(lado,lado)}",
            filetypes=[("Video MP4","*.mp4"),("Todos","*.*")])
        if path:
            destino = {"der": self._path_der, "izq": self._path_izq,
                       "frontal": self._path_frontal}[lado]
            destino.set(path)

    def _log(self, msg):
        def _do():
            self._log_text.insert(tk.END, msg+"\n")
            self._log_text.see(tk.END)
        self.after(0, _do)

    def _set_status(self, msg):
        self.after(0, lambda: self._status_lbl.configure(text=msg))

    def _set_progress(self, pct):
        self.after(0, lambda: self._progress_var.set(pct))

    def _start_processing(self):
        if self._proc_running:
            return

        cm_str = self._cm_var.get().strip()
        hc_cm  = None
        if cm_str:
            try:
                hc_cm = float(cm_str.replace(",","."))
                if hc_cm <= 0: raise ValueError
            except:
                messagebox.showerror("Error", f"Distancia hombro-cadera inválida: '{cm_str}'")
                return

        jobs = []
        orden = [("der", self._path_der), ("izq", self._path_izq), ("frontal", self._path_frontal)]
        for lado, path_var in orden:
            path = path_var.get().strip()
            if not path:
                continue
            if not os.path.exists(path):
                messagebox.showerror("Error", f"El archivo no existe:\n{path}")
                return
            jobs.append((lado, path, hc_cm))

        if not jobs:
            messagebox.showerror("Error","Agregá al menos un video para procesar.")
            return

        self._log_text.delete("1.0", tk.END)
        self._progress_var.set(0)
        self._btn_proc.configure(state="disabled", text="Procesando…")
        self._proc_running = True

        def _run():
            n = len(jobs)
            for job_i, (lado, path, hc_cm) in enumerate(jobs):
                self._set_status(f"Procesando {lado.upper()}  ({job_i+1}/{n})…")
                base = job_i / n * 100
                def prog(pct, base=base, n=n):
                    self._set_progress(base + pct/n)
                try:
                    if lado == "frontal":
                        track, skel, csv = procesar_video_frontal(path, hc_cm, self._log, prog)
                        self._results[lado]["track"]    = track
                        self._results[lado]["skeleton"] = skel
                        self._results[lado]["csv"]      = csv
                        res_izq = self._results.get("izq", {}).get("resultado")
                        segs_izq = res_izq["segmentos"] if res_izq else None
                        if not segs_izq:
                            raise ValueError(
                                "Procesá el video lateral izquierdo (_izq) antes o junto "
                                "con el frontal — el frontal toma sus fases del _izq.")
                        self._log("[FRONTAL] Fases tomadas del perfil izquierdo.\n")
                        resultado = analizar_csv_frontal(csv, segs_der=segs_izq)
                    else:
                        track, skel, csv = procesar_video(path, hc_cm, self._log, prog)
                        self._results[lado]["track"]    = track
                        self._results[lado]["skeleton"] = skel
                        self._results[lado]["csv"]      = csv
                        resultado = analizar_csv(csv, lado)
                    self._results[lado]["resultado"] = resultado
                    self._log(f"[{lado.upper()}] ✔ Listo.\n")
                except Exception as ex:
                    import traceback
                    self._log(f"[{lado.upper()}] ✘ Error: {ex}\n{traceback.format_exc()}\n")
            self.after(0, self._on_proc_done)

        threading.Thread(target=_run, daemon=True).start()

    def _on_proc_done(self):
        self._proc_running = False
        self._btn_proc.configure(state="normal", text="▶  Procesar ambos videos")
        self._set_status("✔ Procesamiento completado.")
        self._set_progress(100)
        self._refresh_vis()

    # ─────────────────────────────────────────────
    # PESTAÑA VISUALIZAR  — layout rediseñado
    # ─────────────────────────────────────────────

    def _build_vis_tab(self):
        v = self._tab_vis

        # ── Barra superior ──
        top = tk.Frame(v, bg=C_PANEL, pady=8)
        top.pack(fill="x")

        mk_lbl(top,"  📊  Visualizador",13,bold=True,color=C_TEXT).pack(side="left",padx=10)

        mk_lbl(top,"  Lado:",10,color=C_SUB).pack(side="left",padx=(20,4))
        for val,txt,col in [("der","Derecho",C_TEXT),("izq","Izquierdo",C_TEXT),
                             ("frontal","Frontal",C_TEXT)]:
            tk.Radiobutton(top,text=txt,variable=self._lado_vis,value=val,
                           command=self._on_lado_change,
                           bg=C_PANEL,fg=col,selectcolor=C_TEXT,
                           activebackground=C_PANEL,font=("Segoe UI",10,"bold")).pack(side="left",padx=2)

        mk_lbl(top,"  Video:",10,color=C_SUB).pack(side="left",padx=(20,4))
        for val,txt in [("track","Trackeado"),("skeleton","Skeleton")]:
            tk.Radiobutton(top,text=txt,variable=self._vidtype_vis,value=val,
                           command=self._on_vidtype_change,
                           bg=C_PANEL,fg=C_TEXT,selectcolor="white",
                           activebackground=C_PANEL,font=("Segoe UI",10)).pack(side="left",padx=2)

        # Botón exportar PDF
        tk.Button(top, text="⬇ Exportar PDF",
                  command=self._export_pdf,
                  bg=C_CARD, fg=C_TEXT, bd=0, padx=12, pady=4,
                  font=("Segoe UI",9,"bold"), cursor="hand2").pack(side="right", padx=12)

        self._avail_lbl = mk_lbl(top,"",10,color=C_SUB)
        self._avail_lbl.pack(side="right",padx=8)

        # ── Cuerpo: columna izquierda (video) + columna derecha (gráfico + datos) ──
        body = tk.Frame(v, bg=C_BG)
        body.pack(fill="both", expand=True, padx=6, pady=4)

        # ── Columna izquierda: video — ancho fijo reducido ──
        left = tk.Frame(body, bg=C_BG, width=380)
        left.pack(side="left", fill="y", expand=False)
        left.pack_propagate(False)

        vid_cont = tk.Frame(left, bg="black")
        vid_cont.pack(fill="both", expand=True, padx=4, pady=4)
        vid_cont.pack_propagate(False)

        self._vid_label = tk.Label(vid_cont, bg="black", bd=0, highlightthickness=0)
        self._vid_label.place(relx=0.5, rely=0.5, anchor="center")
        self._vid_label.image = None

        ctrl = tk.Frame(left, bg=C_PANEL, pady=4)
        ctrl.pack(fill="x", padx=4)
        self._play_btn = tk.Button(ctrl, text="▶ Play", command=self._toggle_play,
                                    bg=C_CARD, fg=C_TEXT, bd=0, padx=10, pady=4,
                                    font=("Segoe UI",9,"bold"), cursor="hand2")
        self._play_btn.pack(side="left", padx=6)
        tk.Button(ctrl,text="⏮ Reiniciar",command=self._reset_video,
                  bg=C_CARD,fg=C_TEXT,bd=0,padx=8,pady=4,
                  font=("Segoe UI",9),cursor="hand2").pack(side="left",padx=4)
        self._slider_var = tk.DoubleVar()
        self._slider = ttk.Scale(ctrl,from_=0,to=100,variable=self._slider_var,command=self._seek)
        self._slider.pack(side="left",fill="x",expand=True,padx=8)
        self._frame_lbl = mk_lbl(ctrl,"0 / 0",9,color=C_SUB)
        self._frame_lbl.pack(side="right",padx=8)

        leg = tk.Frame(left,bg=C_BG,pady=2); leg.pack(fill="x",padx=4)
        for col,txt in [(C_GREEN,"■ Parado"),(C_YELLOW,"■ Parando"),
                         (C_ACCENT,"■ Sentándose"),(C_ACCENT2,"■ Sentado")]:
            tk.Label(leg,text=txt,fg=col,bg=C_BG,font=("Segoe UI",8)).pack(side="left",padx=8)
        self._fase_lbl = mk_lbl(left,"Fase: —",11,bold=True,color=C_ACCENT2)
        self._fase_lbl.pack(pady=2)

        # ── Columna derecha: gráfico + datos desplegables (fijo ~50% ancho) ──
        right = tk.Frame(body, bg=C_BG)
        right.pack(side="right", fill="both", expand=True, padx=4)

        # Gráfico grande (más alto)
        self._fig = Figure(figsize=(6, 4), dpi=90, facecolor=C_PANEL)
        self._ax  = self._fig.add_subplot(111)
        self._fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.14)
        self._chart_canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._chart_canvas.get_tk_widget().pack(fill="x", pady=(0,4), padx=2)

        # Datos desplegables — 5 columnas horizontales con scroll vertical
        datos_outer = tk.Frame(right, bg=C_BG)
        datos_outer.pack(fill="both", expand=True)

        cs = tk.Canvas(datos_outer, bg=C_BG, highlightthickness=0)
        sb2 = ttk.Scrollbar(datos_outer, orient="vertical", command=cs.yview)

        # Frame contenedor de las 5 columnas (se expande horizontalmente)
        self._metrics_frame = tk.Frame(cs, bg=C_BG)
        self._metrics_frame.bind("<Configure>",
            lambda e: cs.configure(scrollregion=cs.bbox("all")))

        # Ventana del canvas: siempre al ancho completo del canvas
        self._metrics_win = cs.create_window((0, 0), window=self._metrics_frame, anchor="nw")

        def _on_canvas_resize(event, c=cs):
            c.itemconfig(self._metrics_win, width=event.width)
        cs.bind("<Configure>", _on_canvas_resize)

        cs.configure(yscrollcommand=sb2.set)
        cs.pack(side="left", fill="both", expand=True)
        sb2.pack(side="right", fill="y")

        # Scroll con rueda del mouse
        def _on_mousewheel(event):
            cs.yview_scroll(int(-1*(event.delta/120)), "units")
        cs.bind_all("<MouseWheel>", _on_mousewheel)

    # ─────────────────────────────────────────────
    # LÓGICA VISUALIZADOR
    # ─────────────────────────────────────────────

    def _refresh_vis(self):
        lado  = self._lado_vis.get()
        datos = self._results[lado]
        tiene = datos["track"] is not None
        self._avail_lbl.configure(
            text=f"{'✔' if self._results['der']['track'] else '○'} Der  "
                 f"{'✔' if self._results['izq']['track'] else '○'} Izq  "
                 f"{'✔' if self._results['frontal']['track'] else '○'} Frontal",
            fg=C_TEXT if tiene else C_SUB)
        self._open_video()
        self._render_metrics()
        self._render_chart()

    def _on_lado_change(self):
        self._stop_video()
        self._refresh_vis()

    def _on_vidtype_change(self):
        self._stop_video()
        self._open_video()

    def _open_video(self):
        lado  = self._lado_vis.get()
        tipo  = self._vidtype_vis.get()
        datos = self._results[lado]
        path  = datos["track"] if tipo=="track" else datos["skeleton"]

        if self._cap:
            self._cap.release()
            self._cap = None

        if not path or not os.path.exists(path):
            self._show_placeholder(
                f"Sin video {tipo} para lado {lado.upper()}.\nProcesá el video primero.")
            return

        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            self._show_placeholder("No se pudo abrir el video.")
            return
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps_v = self._cap.get(cv2.CAP_PROP_FPS) or 30
        self._current_frame = 0
        self.after(100, lambda: self._show_frame(0))

    def _show_placeholder(self, msg):
        img  = Image.new("RGB",(640,360),(15,15,30))
        draw = ImageDraw.Draw(img)
        for li, line in enumerate(msg.split("\n")):
            draw.text((30, 30+li*22), line, fill=(140,140,160))
        photo = ImageTk.PhotoImage(img)
        self._vid_label.configure(image=photo)
        self._vid_label.image = photo

    def _show_frame(self, idx):
        if not self._cap: return
        idx = max(0, min(idx, self._total_frames-1))
        self._current_frame = idx
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if not ret or frame is None: return

        resultado = self._results[self._lado_vis.get()]["resultado"]
        if resultado:
            try:
                fases = resultado["fases"]
                if idx < len(fases):
                    fase = fases[idx]
                    col_map = {
                        "parado":      (78,204,163),
                        "parando":     (245,166,35),
                        "sentandose":  (233,69,96),
                        "sentado_ini": (160,160,176),
                        "sentado_fin": (160,160,176),
                        "sentado":     (160,160,176),
                    }
                    col = col_map.get(fase,(255,255,255))
                    cv2.putText(frame, fase.upper(), (10,36),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
                    icons = {
                        "parado":"● Parado", "parando":"↑ Parando",
                        "sentandose":"↓ Sentándose",
                        "sentado_ini":"● Sentado inicial",
                        "sentado_fin":"● Sentado final",
                    }
                    self._fase_lbl.configure(
                        text=f"Fase: {icons.get(fase, fase)}",
                        fg=f"#{col[2]:02x}{col[1]:02x}{col[0]:02x}")
            except: pass

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fh,fw = frame.shape[:2]
        vw = max(self._vid_label.master.winfo_width(), 320)
        vh = max(self._vid_label.master.winfo_height(), 240)
        scale = min(vw/fw, vh/fh)
        nw,nh = int(fw*scale), int(fh*scale)
        frame = cv2.resize(frame,(nw,nh),interpolation=cv2.INTER_AREA)

        photo = ImageTk.PhotoImage(Image.fromarray(frame))
        self._vid_label.configure(image=photo, width=nw, height=nh)
        self._vid_label.image = photo
        self._frame_lbl.configure(text=f"{idx} / {self._total_frames}")
        self._updating_sl = True
        self._slider_var.set((idx/max(self._total_frames-1,1))*100)
        self._updating_sl = False
        self._update_chart_cursor(idx)

    # ─── Video controls ───

    def _toggle_play(self):
        if not self._cap: return
        self._playing = not self._playing
        if self._playing:
            self._play_btn.configure(text="⏸ Pausar")
            self._video_loop()
        else:
            self._play_btn.configure(text="▶ Play")

    def _stop_video(self):
        self._playing = False
        self._play_btn.configure(text="▶ Play")

    def _video_loop(self):
        if not self._playing: return
        nxt = self._current_frame+1
        if nxt >= self._total_frames:
            self._stop_video(); return
        self._show_frame(nxt)
        self.after(int(1000/self._fps_v), self._video_loop)

    def _reset_video(self):
        self._stop_video()
        self._show_frame(0)

    def _seek(self, val):
        if self._updating_sl or self._total_frames<=0: return
        self._show_frame(int(float(val)/100*(self._total_frames-1)))

    # ─── Secciones desplegables ───

    def _toggle_section(self, clave):
        visible = self._section_visible.get(clave, True)
        nuevo   = not visible
        self._section_visible[clave] = nuevo
        frame = self._section_frames.get(clave)
        btn   = self._section_btns.get(clave)
        if frame:
            if nuevo:
                frame.pack(fill="x", padx=0, pady=(0,4))
            else:
                frame.pack_forget()
        if btn:
            arrow = "▼" if nuevo else "▶"
            cur = btn.cget("text")
            parts = cur.split(" ", 1)
            if len(parts) == 2:
                btn.configure(text=f"{arrow} {parts[1]}")

    def _make_section(self, parent, clave, titulo, color_header):
        """
        Crea una sección con cabecera clicable (toggle) y frame de contenido.
        Devuelve el frame de contenido donde se pueden agregar widgets.
        """
        # Botón-cabecera
        btn = tk.Button(
            parent,
            text=f"▼  {titulo}",
            command=lambda k=clave: self._toggle_section(k),
            bg=color_header, fg=C_TEXT, bd=0,
            padx=8, pady=8,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2", anchor="w", relief="flat",
        )
        btn.pack(fill="x", padx=0, pady=(4,0))
        self._section_btns[clave] = btn

        # Frame contenido
        content = tk.Frame(parent, bg=C_CARD)
        content.pack(fill="x", padx=0, pady=(0,4))
        self._section_frames[clave]  = content
        self._section_visible[clave] = True
        return content

    # ─── Métricas ───

    def _render_metrics(self):
        for w in self._metrics_frame.winfo_children():
            w.destroy()
        self._section_frames.clear()
        self._section_btns.clear()
        self._section_visible.clear()

        lado      = self._lado_vis.get()
        resultado = self._results[lado]["resultado"]

        if not resultado:
            mk_lbl(self._metrics_frame,
                   "Sin datos.\nProcesá el video primero.",
                   11, color=C_SUB).pack(pady=30)
            return

        ORDEN = [
            ("sentado_ini", "Sentado inicial", "#555577"),
            ("parando",     "Parándose",       "#c48a00"),
            ("parado",      "Parado",          "#2a9d7a"),
            ("sentandose",  "Sentándose",      "#b03048"),
            ("sentado_fin", "Sentado final",   "#555577"),
        ]

        cols = {}
        for i, (clave, titulo, _) in enumerate(ORDEN):
            col = tk.Frame(self._metrics_frame, bg=C_BG)
            col.grid(row=0, column=i, sticky="nsew", padx=2, pady=2)
            cols[clave] = col
        for i in range(len(ORDEN)):
            self._metrics_frame.columnconfigure(i, weight=1)
        self._metrics_frame.rowconfigure(0, weight=1)

        sub_counter = [0]

        def make_subsection(parent, titulo_sub, color_sub=C_PANEL):
            # Contenedor externo fijo: agrupa btn + content y nunca se mueve
            slot = tk.Frame(parent, bg=parent.cget("bg"))
            slot.pack(fill="x", padx=4, pady=(4,0))

            btn = tk.Button(
                slot, text=f"▼ {titulo_sub}",
                bg=color_sub, fg=C_TEXT, bd=0,
                padx=6, pady=3,
                font=("Segoe UI", 8, "bold"),
                cursor="hand2", anchor="w", relief="flat"
            )
            btn.pack(fill="x")

            # content_sub vive dentro del slot, siempre después del btn
            content_sub = tk.Frame(slot, bg=C_CARD)
            content_sub.pack(fill="x", pady=(0,2))

            def toggle(b=btn, c=content_sub):
                if c.winfo_ismapped():
                    c.pack_forget()
                    b.configure(text="▶" + b.cget("text")[1:])
                else:
                    # pack dentro del slot → siempre queda debajo del btn
                    c.pack(fill="x", pady=(0,2))
                    b.configure(text="▼" + b.cget("text")[1:])
            btn.configure(command=toggle)
            return content_sub

        for clave, titulo, color_h in ORDEN:
            items = resultado.get(clave, [])
            if not items:
                continue

            content = self._make_section(cols[clave], clave, titulo, color_h)

            for ep_i, m in enumerate(items):
                ep_hdr = tk.Frame(content, bg=color_h)
                ep_hdr.pack(fill="x", padx=0, pady=(4,0))
                mk_lbl(ep_hdr,
                       f"  Ep.{ep_i+1}  "
                       f"f{m['frame_ini']}→{m['frame_fin']}  "
                       f"({m['duracion_s']:.1f}s)",
                       9, bold=True, color=C_TEXT).pack(side="left", pady=3, padx=4)

                if lado == "frontal":
                    unit_x = m.get("dist_x_unit", "")
                    body = tk.Frame(content, bg=C_CARD)
                    body.pack(fill="x", padx=0, pady=(0,4))

                    sub_dy_media = make_subsection(content, "Dif. Y media (der-izq)")
                    for campo, label in [
                        ("hombro","Hombro"), ("mano","Mano"),
                        ("cadera","Cadera"), ("oreja","Oreja"),
                    ]:
                        metric_row(sub_dy_media, label, m.get(f"diff_y_{campo}_media", np.nan), "", C_TEXT)

                    sub_dy_max = make_subsection(content, "Dif. Y máxima (der-izq)")
                    for campo, label in [
                        ("hombro","Hombro"), ("mano","Mano"),
                        ("cadera","Cadera"), ("oreja","Oreja"),
                    ]:
                        metric_row(sub_dy_max, label, m.get(f"diff_y_{campo}_max", np.nan), "", C_TEXT)

                    sub_dx = make_subsection(content, "Distancia X entre manos")
                    if clave == "sentado_ini":
                        metric_row(sub_dx, "Inicial", m.get("dist_x_manos_inicial", np.nan), unit_x, C_TEXT)
                    elif clave == "sentado_fin":
                        metric_row(sub_dx, "Final", m.get("dist_x_manos_final", np.nan), unit_x, C_TEXT)
                    else:
                        metric_row(sub_dx, "Media",  m.get("dist_x_manos_media", np.nan), unit_x, C_TEXT)
                        metric_row(sub_dx, "Máxima", m.get("dist_x_manos_max",   np.nan), unit_x, C_TEXT)

                elif clave in ("sentado_ini", "sentado_fin"):
                    body = tk.Frame(content, bg=C_CARD)
                    body.pack(fill="x", padx=0, pady=(0,4))
                    for ang, label in [
                        ("tobillo","Tobillo"), ("rodilla","Rodilla"), ("cadera","Cadera"),
                        ("brazo","Brazo"), ("cuello_vertical","Cuello-vert"),
                        ("cabeza_cuello","Cabeza-cuello"),
                    ]:
                        metric_row(body, label, m.get(f"{ang}_ini", np.nan), "°", C_TEXT)

                elif clave in ("parando", "sentandose"):
                    sub2 = make_subsection(content, "Mínimos")
                    for ang, label in [
                        ("cuello_vertical","Cuello-vert"), ("cabeza_cuello","Cabeza-cuello"),
                        ("rodilla","Rodilla"), ("cadera","Cadera"), ("tobillo","Tobillo"),
                    ]:
                        metric_row(sub2, label, m.get(f"{ang}_min", np.nan), "°", C_TEXT)

                    sub3 = make_subsection(content, "Máximos")
                    for ang, label in [
                        ("cuello_vertical","Cuello-vert"), ("cabeza_cuello","Cabeza-cuello"),
                        ("brazo","Brazo"),
                    ]:
                        metric_row(sub3, label, m.get(f"{ang}_max", np.nan), "°", C_TEXT)

                    sub4 = make_subsection(content, "ROM")
                    for ang, label in [
                        ("tobillo","Tobillo"), ("rodilla","Rodilla"), ("cadera","Cadera"),
                        ("brazo","Brazo"), ("cuello_vertical","Cuello-vert"),
                        ("cabeza_cuello","Cabeza-cuello"),
                    ]:
                        metric_row(sub4, label, m.get(f"{ang}_rom", np.nan), "°", C_TEXT)

                    sub5 = make_subsection(content, "Vel. angular brazo")
                    metric_row(sub5, "Mediana", m.get("vel_ang_brazo_mediana", np.nan), "°/s", C_TEXT)
                    metric_row(sub5, "Máxima",  m.get("vel_ang_brazo_max",     np.nan), "°/s", C_TEXT)

                    sub6 = make_subsection(content, "Vel. angular cadera")
                    metric_row(sub6, "Mediana", m.get("vel_ang_cadera_mediana", np.nan), "°/s", C_TEXT)
                    metric_row(sub6, "Máxima",  m.get("vel_ang_cadera_max",     np.nan), "°/s", C_TEXT)

                elif clave == "parado":
                    body = tk.Frame(content, bg=C_CARD)
                    body.pack(fill="x", padx=0, pady=(0,4))
                    unit_ds = m.get("estabilidad_ds_unit", "")
                    metric_row(body, "DS hombro-X", m.get("estabilidad_ds", np.nan), unit_ds, C_TEXT)

                # ── Subsección Duración (todas las fases) ──
                sub_dur = make_subsection(content, "Duración")
                fps_vid = 30
                metric_row(sub_dur, "Frames",   m.get("n_frames",   np.nan), "",  C_TEXT)
                metric_row(sub_dur, "Duración", m.get("duracion_s", np.nan), "s", C_TEXT)

                tk.Frame(content, bg=C_BG, height=3).pack()

    # ─── Gráfico ───

    def _render_chart(self):
        self._ax.clear()
        self._ax.set_facecolor(C_PANEL)
        self._fig.patch.set_facecolor(C_PANEL)
        self._ax.tick_params(colors=C_SUB, labelsize=7)
        for sp in self._ax.spines.values(): sp.set_color(C_CARD)
        self._chart_cursor = None

        lado      = self._lado_vis.get()
        resultado = self._results[lado]["resultado"]

        if not resultado:
            self._chart_canvas.draw(); return

        df     = resultado["df"]
        frames = np.arange(len(df))

        if lado == "frontal":
            series = [
                ("cadera_der_y_norm",  "#f5a623", "Cadera der (Y)"),
                ("cadera_izq_y_norm",  "#4ecca3", "Cadera izq (Y)"),
                ("hombro_der_y_norm",  "#a78bfa", "Hombro der (Y)"),
                ("hombro_izq_y_norm",  "#00c8ff", "Hombro izq (Y)"),
            ]
            titulo_chart = "Alturas (Y) Der/Izq — Frontal"
        else:
            series = [
                ("angulo_rodilla",         "#4ecca3", "Rodilla"),
                ("angulo_cadera",          "#f5a623", "Cadera"),
                ("angulo_tobillo",         "#a78bfa", "Tobillo"),
                ("angulo_cuello_vertical", "#00c8ff", "Cuello-vert"),
                ("angulo_cabeza_cuello",   "#ffff55", "Cabeza-cuello"),
            ]
            titulo_chart = f"Ángulos — Lado {lado.upper()}"

        for col, color, label in series:
            if col in df.columns:
                self._ax.plot(frames, df[col].values, color=color, lw=1.4, label=label)

        col_map = {
            "parado":      "#4ecca3",
            "parando":     "#f5a623",
            "sentandose":  "#e94560",
            "sentado_ini": "#aaaaaa",
            "sentado_fin": "#aaaaaa",
            "sentado":     "#aaaaaa",
        }
        for ini, fin, fase in resultado["segmentos"]:
            self._ax.axvspan(ini, fin, alpha=0.18,
                             color=col_map.get(fase, "#ffffff"))

        self._ax.set_xlabel("Frame",      color=C_SUB,  fontsize=7)
        self._ax.set_ylabel("Ángulo (°)" if lado != "frontal" else "Y normalizado",
                            color=C_SUB,  fontsize=7)
        self._ax.set_title(titulo_chart,
                           color=C_TEXT, fontsize=9)
        self._ax.legend(fontsize=6, facecolor=C_CARD,
                        edgecolor=C_CARD, labelcolor=C_TEXT, ncol=2)

        self._chart_cursor = self._ax.axvline(
            x=0, color="white", lw=1.2, ls="--", alpha=0.8)

        self._chart_canvas.draw()

    def _update_chart_cursor(self, frame_idx):
        if self._chart_cursor is None:
            return
        resultado = self._results[self._lado_vis.get()]["resultado"]
        if not resultado:
            return
        self._chart_cursor.set_xdata([frame_idx, frame_idx])
        self._chart_canvas.draw_idle()

    # ─── Exportar PDF ───

    def _export_pdf(self):
        lado      = self._lado_vis.get()
        resultado = self._results[lado]["resultado"]
        if not resultado:
            messagebox.showwarning("Sin datos",
                "No hay datos cargados para exportar.\nProcesá un video primero.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Guardar informe PDF",
            defaultextension=".pdf",
            filetypes=[("PDF","*.pdf")],
            initialfile=f"informe_STS_{lado}.pdf"
        )
        if not save_path:
            return

        try:
            generar_pdf(resultado, lado, self._fig, save_path)
            messagebox.showinfo("PDF generado",
                f"Informe guardado en:\n{save_path}")
        except Exception as ex:
            messagebox.showerror("Error al generar PDF", str(ex))

    # ─────────────────────────────────────────────
    # CIERRE
    # ─────────────────────────────────────────────

    def on_close(self):
        self._stop_video()
        if self._cap: self._cap.release()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
