import cv2
import numpy as np
from typing import Dict, Tuple
import math
import json

class DetectorTablero:
    def __init__(self, config_path='config_tablero.json'):
        # Cargar configuración
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Configurar ROIs
        self.roi = {k: tuple(v) for k, v in config['rois'].items()}
        
        # Configurar rangos de color
        self.rango_luces = config['rangos_color']
        
        # Configurar parámetros del velocímetro
        params_velocimetro = config['velocimetro']
        self.centro_velocimetro = tuple(params_velocimetro['centro'])
        self.radio_velocimetro = params_velocimetro['radio']
        self.angulo_inicio = params_velocimetro['angulo_inicio']
        self.angulo_fin = params_velocimetro['angulo_fin']
        self.rpm_max = 7000  # RPM máximas del velocímetro
        
        # Valores RGB para el freno de mano
        # Interior: RGB(227, 201, 210)
        # Exterior: RGB(220, 96, 109)
        # self.rgb_freno_mano_interior = np.array([220, 220, 220])
        self.rgb_freno_mano_interior = np.array([0, 134, 233])
        self.rgb_freno_mano_exterior = np.array([178, 255, 255])
        # self.rgb_freno_mano_exterior = np.array([220, 96, 109])
        
        # Valores RGB para los giros
        # Interior: RGB(154, 185, 41)
        # Exterior: RGB(255, 250, 250)
        self.rgb_giro_interior = np.array([100, 180, 40])
        self.rgb_giro_exterior = np.array([255, 250, 250])
        
        # Umbral de tolerancia para la detección
        self.umbral_tolerancia = 25

    def detectar_luz(self, frame: np.ndarray, roi: Tuple[int, int, int, int], 
                    rango_color: Tuple[np.ndarray, np.ndarray]) -> bool:
        """
        Detecta si una luz está encendida en una región específica
        """
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(rango_color[0]), 
                          np.array(rango_color[1]))
        return np.mean(mask) > 50

    def detectar_aguja(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> float:
        """
        Detecta la posición de la aguja del velocímetro y calcula las RPM
        """
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        
        # Aplicar umbral para detectar la aguja
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Detectar líneas usando transformada de Hough
        lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 50, 
                               minLineLength=20, maxLineGap=10)
        
        if lines is not None:
            # Encontrar la línea más larga (probablemente la aguja)
            linea_mas_larga = max(lines, key=lambda x: 
                                math.sqrt((x[0][2]-x[0][0])**2 + 
                                        (x[0][3]-x[0][1])**2))
            
            # Calcular el ángulo de la línea
            dx = linea_mas_larga[0][2] - linea_mas_larga[0][0]
            dy = linea_mas_larga[0][3] - linea_mas_larga[0][1]
            angulo = math.degrees(math.atan2(dy, dx))
            
            # Normalizar el ángulo al rango del velocímetro
            angulo_norm = (angulo - self.angulo_inicio) / (self.angulo_fin - self.angulo_inicio)
            angulo_norm = max(0, min(1, angulo_norm))  # Clamp entre 0 y 1
            
            # Convertir a RPM
            rpm = angulo_norm * self.rpm_max
            return rpm
        
        return 0.0

    def detectar_freno_mano(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> bool:
        """
        Detecta si el freno de mano está activado usando los valores RGB específicos
        """
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]
        
        # Crear máscaras para los colores interior y exterior
        mask_interior = cv2.inRange(roi_frame, 
                                   self.rgb_freno_mano_interior - self.umbral_tolerancia,
                                   self.rgb_freno_mano_interior + self.umbral_tolerancia)
        
        mask_exterior = cv2.inRange(roi_frame, 
                                   self.rgb_freno_mano_exterior - self.umbral_tolerancia,
                                   self.rgb_freno_mano_exterior + self.umbral_tolerancia)
        
        # Combinar las máscaras
        mask_combined = cv2.bitwise_or(mask_interior, mask_exterior)
        
        # Calcular el porcentaje de píxeles que coinciden con los colores del freno
        porcentaje_coincidencia = np.sum(mask_combined > 0) / (w * h)
        
        # Debug: mostrar la máscara
        cv2.imshow('Máscara Freno de Mano', mask_combined)
        
        return porcentaje_coincidencia > 0.001  # Umbral ajustable

    def detectar_giro(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> bool:
        """
        Detecta si un giro está activado usando los valores RGB específicos
        """
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]
        
        # Crear máscaras para los colores interior y exterior
        mask_interior = cv2.inRange(roi_frame, 
                                   self.rgb_giro_interior - self.umbral_tolerancia,
                                   self.rgb_giro_interior + self.umbral_tolerancia)
        
        mask_exterior = cv2.inRange(roi_frame, 
                                   self.rgb_giro_exterior - self.umbral_tolerancia,
                                   self.rgb_giro_exterior + self.umbral_tolerancia)
        
        # Combinar las máscaras
        mask_combined = cv2.bitwise_or(mask_interior, mask_exterior)
        
        # Calcular el porcentaje de píxeles que coinciden con los colores del giro
        porcentaje_coincidencia = np.sum(mask_combined > 0) / (w * h)
        
        # Debug: mostrar la máscara
        cv2.imshow('Máscara Giro', mask_combined)
        if porcentaje_coincidencia > 0:
            print(porcentaje_coincidencia) 
        return porcentaje_coincidencia > 0.0001  # Umbral ajustable

    def procesar_frame(self, frame: np.ndarray) -> Dict:
        """
        Procesa un frame y retorna el estado de todos los elementos
        """
        resultados = {}
        
        # Dibujar los ROIs en el frame
        frame_con_rois = frame.copy()
        colores_roi = {
            'freno_mano': (0, 0, 255),      # Rojo
            'luces_altas': (255, 0, 0),     # Azul
            'luces_bajas': (0, 255, 255),   # Amarillo
            'rpm': (255, 0, 255),           # Magenta
            'giro_izquierda': (0, 255, 0),  # Verde
            'giro_derecha': (0, 255, 0)     # Verde
        }
        
        for elemento, roi in self.roi.items():
            x, y, w, h = roi
            cv2.rectangle(frame, (x, y), (x+w, y+h), colores_roi.get(elemento, (255, 255, 255)), 2)
            # Mostrar etiqueta del ROI
            cv2.putText(frame, elemento, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, colores_roi.get(elemento, (255, 255, 255)), 1)
        
        # Detectar luces usando los rangos de color del archivo de configuración
        for elemento, rango in self.rango_luces.items():
            resultados[elemento] = self.detectar_luz(frame, self.roi[elemento], rango)
        
        # Detectar RPM usando la aguja
        resultados['rpm'] = self.detectar_aguja(frame, self.roi['rpm'])
        
        return resultados

def main():
    # Inicializar la cámara o abrir el video
    video_path = "video_completo.MOV"
    cap = cv2.VideoCapture(video_path)
    detector = DetectorTablero()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Procesar el frame
        resultados = detector.procesar_frame(frame)
        
        # Mostrar resultados en pantalla
        y_pos = 30
        for elemento, valor in resultados.items():
            if elemento == 'rpm':
                texto = f"RPM: {valor:.0f}"
            else:
                texto = f"{elemento}: {'ON' if valor else 'OFF'}"
            cv2.putText(frame, texto, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            y_pos += 30
        
        # Mostrar el frame
        cv2.imshow('Detector de Tablero', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 