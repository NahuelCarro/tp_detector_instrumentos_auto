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
        
        # Configurar ROIs normalizados
        self.roi_norm = {k: tuple(v) for k, v in config['rois'].items()}
        
        # Configurar rangos de color
        self.rango_luces = config['rangos_color']
        
        # Configurar parámetros normalizados del velocímetro
        params_velocimetro = config['velocimetro']
        self.centro_velocimetro_norm = tuple(params_velocimetro['centro'])
        self.radio_velocimetro_norm = params_velocimetro['radio']
        self.angulo_inicio = params_velocimetro['angulo_inicio']
        self.angulo_fin = params_velocimetro['angulo_fin']
        self.rpm_max = 8000
        
        # Umbral de tolerancia para la detección
        self.umbral_tolerancia = 25
        
        # Variable para almacenar la última línea de aguja detectada
        self.linea_aguja = None
        
        # Inicializar últimos resultados
        self.ultimos_resultados = None

    def _desnormalizar_roi(
        self, 
        roi_norm: Tuple[float, float, float, float],
        frame_width: int,
        frame_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Desnormaliza un ROI de coordenadas relativas (0-1) a píxeles.
        
        Args:
            roi_norm: ROI normalizado (x, y, w, h) con valores entre 0 y 1
            frame_width: Ancho del frame en píxeles
            frame_height: Alto del frame en píxeles
            
        Returns:
            Tupla con coordenadas en píxeles (x, y, w, h)
        """
        x_norm, y_norm, w_norm, h_norm = roi_norm
        x = int(x_norm * frame_width)
        y = int(y_norm * frame_height)
        w = int(w_norm * frame_width)
        h = int(h_norm * frame_height)
        
        # Asegurar que el ROI sea válido
        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)
        
        # Ajustar si se sale del frame
        if x + w > frame_width:
            w = frame_width - x
        if y + h > frame_height:
            h = frame_height - y
            
        return x, y, w, h
        
    def _desnormalizar_punto(
        self, 
        punto_norm: Tuple[float, float],
        frame_width: int,
        frame_height: int
    ) -> Tuple[int, int]:
        """
        Desnormaliza un punto de coordenadas relativas (0-1) a píxeles.
        
        Args:
            punto_norm: Coordenadas normalizadas (x, y) con valores entre 0 y 1
            frame_width: Ancho del frame en píxeles
            frame_height: Alto del frame en píxeles
            
        Returns:
            Tupla con coordenadas en píxeles (x, y)
        """
        x_norm, y_norm = punto_norm
        x = int(x_norm * frame_width)
        y = int(y_norm * frame_height)
        return x, y

    def detectar_luz(
        self,
        frame: np.ndarray,
        roi_norm: Tuple[float, float, float, float],
        rango_color: Tuple[np.ndarray, np.ndarray]
    ) -> bool:
        """
        Detecta si una luz está encendida en una región específica.

        Args:
            frame: Frame completo
            roi_norm: ROI normalizado (x, y, w, h)
            rango_color: Rango de color HSV (min, max)

        Returns:
            True si se detecta la luz encendida, False en caso contrario
        """
        # Desnormalizar ROI
        frame_height, frame_width = frame.shape[:2]
        x, y, w, h = self._desnormalizar_roi(roi_norm, frame_width, frame_height)
        
        # Comprobar si el ROI tiene tamaño válido después de ajustes
        if w <= 0 or h <= 0:
            return False 
            
        # Extraer ROI absoluto
        roi_frame = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array(rango_color[0]),
            np.array(rango_color[1])
        )
        return np.mean(mask) > 50

    def detectar_aguja(
        self, 
        frame: np.ndarray, 
        roi_norm: Tuple[float, float, float, float]
    ) -> float:
        """
        Detecta la posición de la aguja del velocímetro.

        Args:
            frame: Frame completo
            roi_norm: ROI normalizado del velocímetro (x, y, w, h)

        Returns:
            RPM calculada a partir de la posición de la aguja
        """
        # Desnormalizar ROI
        frame_height, frame_width = frame.shape[:2]
        x, y, w, h = self._desnormalizar_roi(roi_norm, frame_width, frame_height)

        if w <= 0 or h <= 0:
            self.linea_aguja = None
            return 0.0
            
        roi_frame = frame[y:y+h, x:x+w]
        
        # Desnormalizar parámetros del velocímetro
        centro_x_abs, centro_y_abs = self._desnormalizar_punto(
            self.centro_velocimetro_norm, frame_width, frame_height)
        
        # Usamos frame_width para el radio como en la calibración
        radio_abs = int(self.radio_velocimetro_norm * frame_width)
        
        # Crear una región de interés circular con coords absolutas
        centro_rel = (
            centro_x_abs - x, 
            centro_y_abs - y
        )
        mask_circle = np.zeros_like(roi_frame[:, :, 0])
        cv2.circle(
            mask_circle, 
            centro_rel, 
            radio_abs, 
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

        # Aplicar la máscara circular
        mask_aguja = cv2.bitwise_and(mask_red, mask_circle)

        # Crear máscara para el 70% inferior del círculo
        altura_total = roi_frame.shape[0]
        altura_corte = int(altura_total * 0.3)  # 30% superior

        mask_70percent = mask_aguja.copy()
        mask_70percent[:altura_corte, :] = 0  # El 30% superior queda en negro

        # Aplicar máscara del 70%
        mask_final = cv2.bitwise_and(mask_aguja, mask_70percent)

        # Detectar líneas usando transformada de Hough
        lines = cv2.HoughLinesP(
            mask_final,
            1,
            np.pi/180,
            30,
            minLineLength=radio_abs//3,
            maxLineGap=20
        )

        if lines is None:
            self.linea_aguja = None
            return 0.0
        
        # Filtrar líneas por cercanía al centro
        dist_threshold = radio_abs * 0.5
        lineas_filtradas = []

        for line in lines:
            # Coordenadas relativas al ROI
            x1_rel, y1_rel, x2_rel, y2_rel = line[0]
            
            # Calcular punto más cercano al centro (relativo al ROI)
            px, py = centro_rel
            dx = x2_rel - x1_rel
            dy = y2_rel - y1_rel
            len_squared = dx*dx + dy*dy

            if len_squared == 0:  # Si la línea es un punto
                continue

            # Proyectar el centro en la línea
            t = max(0, min(1, ((px-x1_rel)*dx + (py-y1_rel)*dy) / len_squared))
            # Punto más cercano en la línea (relativo al ROI)
            proj_x = x1_rel + t * dx
            proj_y = y1_rel + t * dy
            # Distancia del centro al punto más cercano (en píxeles del ROI)
            dist = math.sqrt((px-proj_x)**2 + (py-proj_y)**2)

            # Si está dentro del umbral de distancia (en píxeles), la agregamos
            if dist < dist_threshold:
                # Guardamos las coordenadas relativas al ROI
                lineas_filtradas.append(line[0]) 
        
        if not lineas_filtradas:
            self.linea_aguja = None
            return 0.0
        
        # Calcular los vectores directores de cada línea
        vectores = []
        cx_rel, cy_rel = centro_rel  # Centro relativo al ROI
        
        for x1_rel, y1_rel, x2_rel, y2_rel in lineas_filtradas:
            # Vector director de la línea (en coords relativas)
            dx = x2_rel - x1_rel
            dy = y2_rel - y1_rel
            longitud = math.sqrt(dx*dx + dy*dy)

            # Normalizar vector
            if longitud > 0:
                dx /= longitud
                dy /= longitud

            # Asegurar que el vector apunte desde el centro hacia afuera
            # (usando coords relativas)
            dist1 = (x1_rel-cx_rel)**2 + (y1_rel-cy_rel)**2
            dist2 = (x2_rel-cx_rel)**2 + (y2_rel-cy_rel)**2

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
        
        # Crear línea promedio desde el centro (relativo al ROI)
        # Usamos radio absoluto para la longitud
        longitud_linea = radio_abs * 0.8  
        x1_rel_avg = int(cx_rel)
        y1_rel_avg = int(cy_rel)
        x2_rel_avg = int(cx_rel + dx_avg * longitud_linea)
        y2_rel_avg = int(cy_rel + dy_avg * longitud_linea)
        
        # Guardar la línea para dibujarla después
        self.linea_aguja = (
            x1_rel_avg + x, y1_rel_avg + y,
            x2_rel_avg + x, y2_rel_avg + y
        )

        # Calcular el ángulo de la línea
        theta = math.atan2(-dy_avg, -dx_avg)

        # Normalizar el ángulo al rango del velocímetro
        # Convertir de radianes a grados
        angulo = math.degrees(theta)
        
        # Normalizar el ángulo al rango del velocímetro
        rango_angulos = self.angulo_fin - self.angulo_inicio
        # Manejo de posible división por cero o rango inválido
        if rango_angulos == 0:
            angulo_norm = 0.0
        else:
            # Asegurar que el cálculo maneje la vuelta (e.g., 350 a 10 grados)
            angulo_rel = angulo - self.angulo_inicio
            # Ajustar por si cruza 360/0 grados
            if rango_angulos < 0:  # Si fin < inicio (ej: 350 a 10)
                angulo_rel = (angulo_rel + 360) % 360
                rango_angulos += 360
            elif angulo_rel < 0:  # Si angulo < inicio
                angulo_rel += 360
            
            angulo_norm = angulo_rel / rango_angulos
            angulo_norm = max(0, min(1, angulo_norm))  # Clamp entre 0 y 1
        
        # Convertir a RPM y devolver el valor
        rpm = angulo_norm * self.rpm_max

        return rpm

    def detectar_estado_instrumentos(self, frame: np.ndarray) -> Dict:
        """
        Detecta el estado de todos los instrumentos.
        """
        resultados = {}

        # Detectar luces usando los rangos de color del archivo de configuración
        for elemento, rango in self.rango_luces.items():
            if elemento != 'rpm':  # El rpm se maneja diferente
                resultados[elemento] = self.detectar_luz(
                    frame, self.roi_norm[elemento], rango
                )

        # Detectar RPM usando la aguja
        resultados['rpm'] = self.detectar_aguja(frame, self.roi_norm['rpm'])

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
        """Dibuja los rectángulos de las regiones de interés normalizadas."""
        colores_roi = {
            'freno_mano': (0, 0, 255),      # Rojo
            'luces_altas': (255, 0, 0),     # Azul
            'luces_bajas': (0, 255, 0),     # Verde
            'rpm': (255, 0, 255),           # Magenta
            'giro_izquierda': (0, 255, 0),  # Verde
            'giro_derecha': (0, 255, 0)     # Verde
        }
        
        frame_height, frame_width = frame.shape[:2]
        
        for elemento, roi_norm in self.roi_norm.items():
            # Desnormalizar ROI
            x, y, w, h = self._desnormalizar_roi(roi_norm, frame_width, frame_height)
            
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
        """Dibuja el centro, radio y aguja detectada (si existe)."""
        frame_height, frame_width = frame.shape[:2]
        
        # Desnormalizar centro y radio para dibujar
        centro_x_abs, centro_y_abs = self._desnormalizar_punto(
            self.centro_velocimetro_norm, frame_width, frame_height)
        centro_abs = (centro_x_abs, centro_y_abs)
        radio_abs = int(self.radio_velocimetro_norm * frame_width)

        # Dibujar el centro del velocímetro
        cv2.circle(frame, centro_abs, 5, (0, 0, 255), -1)
        # Dibujar el círculo del velocímetro
        cv2.circle(
            frame, 
            centro_abs, 
            radio_abs, 
            (0, 255, 0),
            2
        )

        # Dibujar la línea de la aguja (ya está en coordenadas globales)
        if self.linea_aguja is not None:
            x1, y1, x2, y2 = self.linea_aguja
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

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
    video_path = "video_completo.mov"
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