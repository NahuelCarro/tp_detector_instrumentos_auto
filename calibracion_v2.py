import cv2
import numpy as np
import json
import os

def calibrar_roi(frame, nombre_elemento):
    """Permite al usuario seleccionar una región de interés"""
    roi = cv2.selectROI(f"Selecciona ROI para {nombre_elemento}", frame, False)
    cv2.destroyWindow(f"Selecciona ROI para {nombre_elemento}")
    return list(roi)  # Convertir tupla a lista


def main():
    # Abrir video
    video_path = "video_completo.MOV"
    cap = cv2.VideoCapture(video_path)
    
    # Leer primer frame
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: No se pudo leer el primer frame del video")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Calibrar ROIs
    print("Calibrando regiones de interés (ROIs)...")
    rois = {}
    elementos = ['tablero']
    
    for elemento in elementos:
        print(f"Selecciona la región para {elemento}")
        roi = calibrar_roi(frame, elemento)
        if roi is None:
            print(f"Error al calibrar ROI para {elemento}")
            cap.release()
            cv2.destroyAllWindows()
            return
        rois[elemento] = roi
    
    
    
    # Guardar configuración
    config = {
        'rois': rois,
    }
    
    try:
        with open('config_tablero_completo.json', 'w') as f:
            json.dump(config, f, indent=4)
        print("Calibración completada. Configuración guardada en 'config_tablero_completo.json'")
    except Exception as e:
        print(f"Error al guardar la configuración: {e}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 