import cv2
import numpy as np
import json
import os

def calibrar_roi(frame, nombre_elemento):
    """Permite al usuario seleccionar una región de interés"""
    roi = cv2.selectROI(f"Selecciona ROI para {nombre_elemento}", frame, False)
    cv2.destroyWindow(f"Selecciona ROI para {nombre_elemento}")
    return list(roi)  # Convertir tupla a lista

def calibrar_color(frame, nombre_elemento):
    """Permite al usuario seleccionar un rango de color HSV"""
    def nothing(x):
        pass
    
    # Crear ventana para trackbars
    window_name = f'Calibración de color - {nombre_elemento}'
    cv2.namedWindow(window_name)
    
    # Crear trackbars para H, S, V
    cv2.createTrackbar('H_min', window_name, 0, 179, nothing)
    cv2.createTrackbar('S_min', window_name, 0, 255, nothing)
    cv2.createTrackbar('V_min', window_name, 0, 255, nothing)
    cv2.createTrackbar('H_max', window_name, 179, 179, nothing)
    cv2.createTrackbar('S_max', window_name, 255, 255, nothing)
    cv2.createTrackbar('V_max', window_name, 255, 255, nothing)
    
    # Valores iniciales para rojo (común en luces)
    cv2.setTrackbarPos('H_min', window_name, 0)
    cv2.setTrackbarPos('S_min', window_name, 100)
    cv2.setTrackbarPos('V_min', window_name, 100)
    
    while True:
        try:
            # Obtener valores actuales
            h_min = cv2.getTrackbarPos('H_min', window_name)
            s_min = cv2.getTrackbarPos('S_min', window_name)
            v_min = cv2.getTrackbarPos('V_min', window_name)
            h_max = cv2.getTrackbarPos('H_max', window_name)
            s_max = cv2.getTrackbarPos('S_max', window_name)
            v_max = cv2.getTrackbarPos('V_max', window_name)
            
            # Convertir a HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Crear máscara
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            mask = cv2.inRange(hsv, lower, upper)
            
            # Aplicar máscara
            result = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Mostrar resultado
            cv2.imshow(window_name, result)
        except cv2.error as e:
            print(f"Error procesando imagen: {e}")
            break
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyWindow(window_name)
    return ([h_min, s_min, v_min], [h_max, s_max, v_max])

def calibrar_velocimetro(frame):
    """Permite al usuario calibrar el velocímetro"""
    if frame is None:
        print("Error: Frame inválido en calibrar_velocimetro")
        return None
    
    # Seleccionar centro del velocímetro
    print("Haz clic en el centro del velocímetro")
    centro = None
    
    def click_event(event, x, y, flags, param):
        nonlocal centro
        if event == cv2.EVENT_LBUTTONDOWN:
            centro = (x, y)
            cv2.destroyWindow('Selecciona centro del velocímetro')
    
    cv2.namedWindow('Selecciona centro del velocímetro')
    cv2.setMouseCallback('Selecciona centro del velocímetro', click_event)
    
    while centro is None:
        cv2.imshow('Selecciona centro del velocímetro', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if centro is None:
        print("Error: No se seleccionó el centro del velocímetro")
        return None
    
    # Seleccionar radio
    print("Haz clic en el borde del velocímetro para determinar el radio")
    radio = None
    
    def click_event_radio(event, x, y, flags, param):
        nonlocal radio
        if event == cv2.EVENT_LBUTTONDOWN:
            dx = x - centro[0]
            dy = y - centro[1]
            radio = int(np.sqrt(dx*dx + dy*dy))
            cv2.destroyWindow('Selecciona radio del velocímetro')
    
    cv2.namedWindow('Selecciona radio del velocímetro')
    cv2.setMouseCallback('Selecciona radio del velocímetro', click_event_radio)
    
    while radio is None:
        temp_frame = frame.copy()
        cv2.circle(temp_frame, centro, 5, (0, 255, 0), -1)
        cv2.imshow('Selecciona radio del velocímetro', temp_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Seleccionar ángulos de inicio y fin
    print("Haz clic en la posición de 0 RPM")
    angulo_inicio = None
    
    def click_event_angulo_inicio(event, x, y, flags, param):
        nonlocal angulo_inicio
        if event == cv2.EVENT_LBUTTONDOWN:
            dx = x - centro[0]
            dy = y - centro[1]
            angulo_inicio = int(np.degrees(np.arctan2(dy, dx)))
            cv2.destroyWindow('Selecciona ángulo de inicio')
    
    cv2.namedWindow('Selecciona ángulo de inicio')
    cv2.setMouseCallback('Selecciona ángulo de inicio', click_event_angulo_inicio)
    
    while angulo_inicio is None:
        temp_frame = frame.copy()
        cv2.circle(temp_frame, centro, radio, (0, 255, 0), 1)
        cv2.imshow('Selecciona ángulo de inicio', temp_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print("Haz clic en la posición de RPM máximas")
    angulo_fin = None
    
    def click_event_angulo_fin(event, x, y, flags, param):
        nonlocal angulo_fin
        if event == cv2.EVENT_LBUTTONDOWN:
            dx = x - centro[0]
            dy = y - centro[1]
            angulo_fin = int(np.degrees(np.arctan2(dy, dx)))
            cv2.destroyWindow('Selecciona ángulo final')
    
    cv2.namedWindow('Selecciona ángulo final')
    cv2.setMouseCallback('Selecciona ángulo final', click_event_angulo_fin)
    
    while angulo_fin is None:
        temp_frame = frame.copy()
        cv2.circle(temp_frame, centro, radio, (0, 255, 0), 1)
        cv2.imshow('Selecciona ángulo final', temp_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    return {
        'centro': list(centro),
        'radio': radio,
        'angulo_inicio': angulo_inicio,
        'angulo_fin': angulo_fin
    }

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
    elementos = ['freno_mano', 'luces_altas', 'luces_bajas', 'rpm', 
                'giro_izquierda', 'giro_derecha']
    
    for elemento in elementos:
        print(f"Selecciona la región para {elemento}")
        roi = calibrar_roi(frame, elemento)
        if roi is None:
            print(f"Error al calibrar ROI para {elemento}")
            cap.release()
            cv2.destroyAllWindows()
            return
        rois[elemento] = roi
    
    # Calibrar colores
    print("Calibrando rangos de color...")
    rangos_color = {}
    elementos_color = ['freno_mano', 'luces_altas', 'luces_bajas', 
                      'giro_izquierda', 'giro_derecha']
    
    for elemento in elementos_color:
        print(f"Calibra el rango de color para {elemento}")
        rangos_color[elemento] = calibrar_color(frame, elemento)
    
    # Calibrar velocímetro
    print("Calibrando velocímetro...")
    params_velocimetro = calibrar_velocimetro(frame)
    if params_velocimetro is None:
        print("Error al calibrar el velocímetro")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Guardar configuración
    config = {
        'rois': rois,
        'rangos_color': rangos_color,
        'velocimetro': params_velocimetro
    }
    
    try:
        with open('config_tablero.json', 'w') as f:
            json.dump(config, f, indent=4)
        print("Calibración completada. Configuración guardada en 'config_tablero.json'")
    except Exception as e:
        print(f"Error al guardar la configuración: {e}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 