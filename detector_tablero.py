import cv2
import numpy as np
from typing import Dict, Tuple
import math
import json


class DetectorTablero:
    def __init__(self, config_path='config_tablero.json'):
        """Inicializa el detector de tablero con la configuración especificada."""
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
        self.rpm_max = 8000  # RPM máximas del velocímetro
        
        # Umbral de tolerancia para la detección
        self.umbral_tolerancia = 25
        
        # Variable para almacenar la última línea de aguja detectada
        self.linea_aguja = None
        
        # Inicializar últimos resultados
        self.ultimos_resultados = None

    def detectar_luz(
        self, 
        frame: np.ndarray, 
        roi: Tuple[int, int, int, int], 
        rango_color: Tuple[np.ndarray, np.ndarray]
    ) -> bool:
        """
        Detecta si una luz está encendida en una región específica.
        
        Args:
            frame: Frame completo
            roi: Región de interés (x, y, ancho, alto)
            rango_color: Rango de color HSV (min, max)
            
        Returns:
            True si se detecta la luz encendida, False en caso contrario
        """
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv, 
            np.array(rango_color[0]), 
            np.array(rango_color[1])
        )
        return np.mean(mask) > 50

    def detectar_aguja(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> float:
        """
        Detecta la posición de la aguja del velocímetro usando filtrado por color
        y método de línea promedio a partir de todas las líneas detectadas.
        
        Args:
            frame: Frame completo
            roi: Región de interés (x, y, ancho, alto)
            
        Returns:
            RPM calculada a partir de la posición de la aguja
        """
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]
        
        # Crear una región de interés circular
        centro_rel = (
            self.centro_velocimetro[0] - x, 
            self.centro_velocimetro[1] - y
        )
        mask_circle = np.zeros_like(roi_frame[:, :, 0])
        cv2.circle(
            mask_circle, 
            centro_rel, 
            self.radio_velocimetro, 
            255, 
            -1
        )
        
        # Convertir a HSV para filtrado por color
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        
        # Obtener rangos de color para la aguja desde la configuración
        lower_red = np.array(self.rango_luces.get('rpm')[0])
        upper_red = np.array(self.rango_luces.get('rpm')[1])
        
        # Crear máscara de color
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        
        # Aplicar la máscara circular para considerar solo el área del velocímetro
        mask_aguja = cv2.bitwise_and(mask_red, mask_circle)
        
        # Crear máscara para el 65% inferior del círculo
        altura_total = roi_frame.shape[0]
        altura_corte = int(altura_total * 0.3)  # 30% superior
        
        mask_65percent = mask_aguja.copy()
        mask_65percent[:altura_corte, :] = 0  # El 30% superior queda en negro
        
        # Aplicar máscara del 65%
        mask_final = cv2.bitwise_and(mask_aguja, mask_65percent)
        
        # Suavizar la máscara para mejorar la detección de líneas
        # kernel = np.ones((10, 10), np.uint8)
        # mask_dilated = cv2.dilate(mask_final, kernel, iterations=1)
        # mask_eroded = cv2.erode(mask_dilated, kernel, iterations=1)
        
        # Detectar líneas usando transformada de Hough
        lines = cv2.HoughLinesP(
            mask_final, 
            1, 
            np.pi/180, 
            30,
            minLineLength=self.radio_velocimetro//3, 
            maxLineGap=20
        )
        
        if lines is None:
            self.linea_aguja = None
            return 0.0
        
        # Filtrar líneas por cercanía al centro
        dist_threshold = self.radio_velocimetro * 0.5
        lineas_filtradas = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calcular punto más cercano al centro en la línea
            px, py = centro_rel
            dx = x2 - x1
            dy = y2 - y1
            len_squared = dx*dx + dy*dy
            
            if len_squared == 0:  # Si la línea es un punto
                continue
                
            # Proyectar el centro en la línea
            t = max(0, min(1, ((px-x1)*dx + (py-y1)*dy) / len_squared))
            # Punto más cercano en la línea
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            # Distancia del centro al punto más cercano
            dist = math.sqrt((px-proj_x)**2 + (py-proj_y)**2)
            
            # Si está dentro del umbral de distancia, la agregamos
            if dist < dist_threshold:
                lineas_filtradas.append(line[0])
        
        if not lineas_filtradas:
            self.linea_aguja = None
            return 0.0
        
        # Calcular los vectores directores de cada línea
        # y normalizarlos para que estén orientados desde el centro hacia afuera
        vectores = []
        cx, cy = centro_rel
        
        for x1, y1, x2, y2 in lineas_filtradas:
            # Vector director de la línea
            dx = x2 - x1
            dy = y2 - y1
            longitud = math.sqrt(dx*dx + dy*dy)
            
            # Normalizar vector
            if longitud > 0:
                dx /= longitud
                dy /= longitud
            
            # Asegurar que el vector apunte desde el centro hacia afuera
            # Determinar cuál extremo está más lejos del centro
            dist1 = (x1-cx)**2 + (y1-cy)**2
            dist2 = (x2-cx)**2 + (y2-cy)**2
            
            # Si el primer punto está más lejos del centro, invertir dirección
            if dist1 > dist2:
                dx = -dx
                dy = -dy
            
            vectores.append((dx, dy))
        
        # Calcular el vector promedio
        if not vectores:
            self.linea_aguja = None
            return 0.0
        
        dx_sum = sum(v[0] for v in vectores)
        dy_sum = sum(v[1] for v in vectores)
        n = len(vectores)
        dx_avg = dx_sum / n
        dy_avg = dy_sum / n
        
        # Normalizar el vector promedio
        norm = math.sqrt(dx_avg**2 + dy_avg**2)
        if norm > 0:
            dx_avg /= norm
            dy_avg /= norm
        
        # Crear línea promedio desde el centro
        longitud_linea = self.radio_velocimetro * 0.8  # Longitud deseada
        x1 = int(cx)
        y1 = int(cy)
        x2 = int(cx + dx_avg * longitud_linea)
        y2 = int(cy + dy_avg * longitud_linea)
        
        # Guardar la línea para dibujarla después
        self.linea_aguja = (
            x1 + x, y1 + y,  # punto inicial en coordenadas globales
            x2 + x, y2 + y   # punto final en coordenadas globales
        )
        
        # Calcular el ángulo de la línea usando la misma convención que el código original
        # para asegurar compatibilidad con la fórmula de RPM
        theta = math.atan2(-dy_avg, -dx_avg)
        
        # Normalizar el ángulo al rango del velocímetro
        # Convertir de radianes a grados
        angulo = math.degrees(theta)
        
        # Normalizar el ángulo al rango del velocímetro
        rango_angulos = self.angulo_fin - self.angulo_inicio
        angulo_norm = (angulo - self.angulo_inicio) / rango_angulos
        angulo_norm = max(0, min(1, angulo_norm))  # Clamp entre 0 y 1
        
        # Convertir a RPM y devolver el valor
        rpm = angulo_norm * self.rpm_max
        
        return rpm

    def detectar_estado_instrumentos(self, frame: np.ndarray) -> Dict:
        """
        Detecta el estado de todos los instrumentos y retorna un diccionario 
        con los resultados.
        """
        resultados = {}
        
        # Detectar luces usando los rangos de color del archivo de configuración
        for elemento, rango in self.rango_luces.items():
            if elemento != 'rpm':  # El rpm se maneja diferente
                resultados[elemento] = self.detectar_luz(
                    frame, self.roi[elemento], rango
                )
        
        # Detectar RPM usando la aguja
        resultados['rpm'] = self.detectar_aguja(frame, self.roi['rpm'])
        
        # Guardar los resultados
        self.ultimos_resultados = resultados
        
        return resultados
        
    def dibujar_frame(self, frame: np.ndarray, resultados: Dict) -> np.ndarray:
        """
        Dibuja los ROIs y el estado de los instrumentos en el frame.
        
        Args:
            frame: Frame a dibujar
            resultados: Diccionario con los resultados de la detección
            
        Returns:
            Frame con los elementos dibujados
        """
        self._dibujar_rois(frame)
        self._dibujar_elementos_velocimetro(frame)
        self._dibujar_resultados(frame, resultados)
        
        return frame
    
    def _dibujar_rois(self, frame: np.ndarray):
        """Dibuja los rectángulos de las regiones de interés."""
        colores_roi = {
            'freno_mano': (0, 0, 255),      # Rojo
            'luces_altas': (255, 0, 0),     # Azul
            'luces_bajas': (0, 255, 0),     # Verde
            'rpm': (255, 0, 255),           # Magenta
            'giro_izquierda': (0, 255, 0),  # Verde
            'giro_derecha': (0, 255, 0)     # Verde
        }
        
        for elemento, roi in self.roi.items():
            x, y, w, h = roi
            cv2.rectangle(
                frame, 
                (x, y), 
                (x+w, y+h), 
                colores_roi.get(elemento, (255, 255, 255)), 
                2
            )
            # Mostrar etiqueta del ROI
            cv2.putText(
                frame,
                elemento,
                (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, 
                colores_roi.get(elemento, (255, 255, 255)), 
                1
            )

    def _dibujar_elementos_velocimetro(self, frame: np.ndarray):
        """Dibuja el centro del velocímetro y la aguja detectada."""
        if self.linea_aguja is not None:
            x1, y1, x2, y2 = self.linea_aguja
            # Dibujar la línea de la aguja
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            # Dibujar el centro del velocímetro
            cv2.circle(frame, self.centro_velocimetro, 5, (0, 0, 255), -1)
            # Dibujar el círculo del velocímetro
            cv2.circle(
                frame, 
                self.centro_velocimetro, 
                self.radio_velocimetro, 
                (0, 255, 0), 
                2
            )
    
    def _dibujar_resultados(self, frame: np.ndarray, resultados: Dict):
        """Dibuja los resultados de la detección en el frame."""
        y_pos = 30
        for elemento, valor in resultados.items():
            if elemento == 'rpm':
                texto = f"RPM: {valor:.0f}"
            else:
                texto = f"{elemento}: {'ON' if valor else 'OFF'}"
            cv2.putText(
                frame, 
                texto, 
                (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX,
                1, 
                (0, 255, 0), 
                2
            )
            y_pos += 30


def main():
    """Función principal para procesar el video y generar el resultado."""
    # Inicializar la cámara o abrir el video
    video_path = "video_completo.MOV"
    cap = cv2.VideoCapture(video_path)
    detector = DetectorTablero()
    
    # Obtener información del video para configurar el VideoWriter
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Configurar el VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "video_detecciones.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Procesando video... El resultado se guardará en {output_path}")
    print("Solo se procesará 1 de cada 15 frames")
    
    # Contador de frames
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detectar estado de instrumentos solo cada 15 frames o al inicio
        if frame_count % 15 == 0 or detector.ultimos_resultados is None:
            print(f"Detectando instrumentos en frame {frame_count}")
            resultados = detector.detectar_estado_instrumentos(frame)
        else:
            # Usar los últimos resultados disponibles
            resultados = detector.ultimos_resultados
        
        # Dibujar en todos los frames
        detector.dibujar_frame(frame, resultados)
        
        # Guardar el frame en el video de salida
        out.write(frame)
        
        # Mostrar el frame
        cv2.imshow('Detector de Tablero', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    print(f"Video guardado en: {output_path}")
    print(f"Total de frames procesados: {frame_count}")
    print(f"Detecciones realizadas: {frame_count // 15}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()