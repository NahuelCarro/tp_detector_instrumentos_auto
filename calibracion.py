import cv2
import numpy as np
import json


def calibrar_roi(frame, nombre_elemento):
    """Permite al usuario seleccionar una región de interés y la devuelve normalizada
    
    Args:
        frame: Frame del video donde se seleccionará la ROI
        nombre_elemento: Nombre del elemento que se está calibrando
        
    Returns:
        Lista con las coordenadas normalizadas de la ROI [x, y, w, h] o None si se canceló
    """
    height, width = frame.shape[:2]
    
    roi = cv2.selectROI(f"Selecciona ROI para {nombre_elemento}", frame, False)
    cv2.destroyWindow(f"Selecciona ROI para {nombre_elemento}")
    
    if roi == (0, 0, 0, 0):  # El usuario presionó ESC o cerró la ventana
        print("Selección de ROI cancelada.")
        return None
        
    x, y, w, h = roi
    
    # Normalizar ROI
    x_norm = round(x / width, 4)
    y_norm = round(y / height, 4)
    w_norm = round(w / width, 4)
    h_norm = round(h / height, 4)
    
    return [x_norm, y_norm, w_norm, h_norm] # Devolver lista normalizada


def calibrar_color(frame, nombre_elemento):
    """Permite al usuario seleccionar un rango de color HSV
    
    Args:
        frame: Frame del video donde se seleccionará el color
        nombre_elemento: Nombre del elemento que se está calibrando
        
    Returns:
        Tupla con dos listas: ([H_min, S_min, V_min], [H_max, S_max, V_max])
    """
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
    """Permite al usuario calibrar el velocímetro
    
    Args:
        frame: Frame del video donde se calibrará el velocímetro
        
    Returns:
        Diccionario con los parámetros del velocímetro:
        {
            'centro': [x_norm, y_norm],  # Coordenadas normalizadas del centro
            'radio': radio_norm,         # Radio normalizado
            'angulo_inicio': angulo_inicio,  # Ángulo de inicio en grados
            'angulo_fin': angulo_fin     # Ángulo final en grados
        }
        o None si hubo un error
    """
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

    cv2.namedWindow('Selecciona ángulo del maximo de RPM')
    cv2.setMouseCallback('Selecciona ángulo del maximo de RPM', 
                         click_event_angulo_inicio)

    while angulo_inicio is None:
        temp_frame = frame.copy()
        cv2.circle(temp_frame, centro, radio, (0, 255, 0), 1)
        cv2.imshow('Selecciona ángulo del maximo de RPM', temp_frame)
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

    cv2.namedWindow('Selecciona ángulo de las 0 RPM')
    cv2.setMouseCallback('Selecciona ángulo de las 0 RPM', click_event_angulo_fin)

    while angulo_fin is None:
        temp_frame = frame.copy()
        cv2.circle(temp_frame, centro, radio, (0, 255, 0), 1)
        cv2.imshow('Selecciona ángulo de las 0 RPM', temp_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Obtener dimensiones para normalizar
    height, width = frame.shape[:2] 

    # Normalizar centro y radio
    if centro is not None and radio is not None:
        centro_norm = [centro[0] / width, centro[1] / height]
        # Normalizamos el radio respecto al ancho (podría ser alto o promedio)
        radio_norm = radio / width 
    else:
        print("Error: No se pudo obtener centro o radio para normalizar")
        return None

    return {
        'centro': centro_norm, # Guardar normalizado
        'radio': radio_norm,    # Guardar normalizado
        'angulo_inicio': angulo_inicio,
        'angulo_fin': angulo_fin
    }


def seleccionar_frame(cap, elemento):
    """Permite al usuario seleccionar un frame específico del video
    
    Args:
        cap: Objeto VideoCapture del video
        elemento: Nombre del elemento que se está calibrando
        
    Returns:
        Frame seleccionado o None si se canceló
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duracion = total_frames / fps

    # Crear una ventana con una trackbar
    window_name = f'Selección de Frame para {elemento}'
    cv2.namedWindow(window_name)

    # Función de callback para la trackbar
    def on_trackbar_change(pos):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        return

    # Crear trackbar
    cv2.createTrackbar('Frame', window_name, 0, total_frames-1, 
                      on_trackbar_change)

    frame_seleccionado = None
    frame_actual = 0

    print(f"Video: {total_frames} frames, {duracion:.2f} segundos")
    print("Usa la barra para navegar por el video")
    print("Presiona 'ESPACIO' para seleccionar el frame actual")
    print("Presiona 'q' para salir")

    while True:
        # Obtener posición actual de la trackbar
        pos = cv2.getTrackbarPos('Frame', window_name)

        # Si cambió la posición, actualizar el frame
        if pos != frame_actual:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            frame_actual = pos

        # Leer frame
        ret, frame = cap.read()
        if not ret:
            break

        # Mostrar tiempo actual
        tiempo_actual = frame_actual / fps
        minutos = int(tiempo_actual / 60)
        segundos = tiempo_actual % 60
        cv2.putText(frame, f"{minutos}:{segundos:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar frame
        cv2.imshow(window_name, frame)

        # Esperar tecla
        key = cv2.waitKey(30) & 0xFF

        # Si se presiona ESPACIO, seleccionar frame actual
        if key == 32:  # ESPACIO
            frame_seleccionado = frame.copy()
            break

        # Si se presiona q, salir
        if key == ord('q'):
            break

    cv2.destroyWindow(window_name)
    return frame_seleccionado


def main():
    """Función principal que ejecuta el proceso de calibración completo
    
    La función:
    1. Abre el video
    2. Para cada elemento a calibrar:
       - Permite seleccionar un frame
       - Calibra la ROI
       - Calibra el rango de color
       - Si es el velocímetro, calibra sus parámetros
    3. Guarda la configuración en un archivo JSON
    """
    # Abrir video
    video_path = "video_completo.MOV"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: No se pudo abrir el video")
        return

    # Definir los elementos a calibrar
    elementos = ['freno_mano', 'luces_altas', 'luces_bajas', 
                 'giro_izquierda', 'giro_derecha', 'rpm'] 
    # elementos = ['rpm']

    # Configuración final
    rois = {}
    rangos_color = {}
    params_velocimetro = None

    # Calibrar cada elemento individualmente
    for elemento in elementos:
        print(f"\n--- Calibrando {elemento} ---")

        # Seleccionar frame para este elemento
        print(f"Selecciona el frame para {elemento}")
        frame = seleccionar_frame(cap, elemento)

        if frame is None:
            print(f"No se seleccionó frame para {elemento}. Saltando...")
            continue

        # Calibrar ROI para este elemento
        print(f"Selecciona la región para {elemento}")
        roi = calibrar_roi(frame, elemento)
        if roi is None:
            print(f"Error al calibrar ROI para {elemento}")
            continue

        rois[elemento] = roi

        # Calibrar color (excepto para RPM)
        print(f"Calibra el rango de color para {elemento}")
        rango_color = calibrar_color(frame, elemento)
        rangos_color[elemento] = rango_color

        # Si es el elemento RPM, calibrar el velocímetro
        if elemento == 'rpm':
            print("Calibrando velocímetro...")
            params_velocimetro = calibrar_velocimetro(frame)
            if params_velocimetro is None:
                print("Error al calibrar el velocímetro")

    # Verificar si se calibró al menos un elemento
    if not rois:
        print("No se calibró ningún elemento. Saliendo...")
        cap.release()
        cv2.destroyAllWindows()
        return

    # Guardar configuración
    config = {
        'rois': rois,
        'rangos_color': rangos_color
    }

    if params_velocimetro:
        config['velocimetro'] = params_velocimetro

    try:
        with open('config_tablero.json', 'w') as f:
            json.dump(config, f, indent=4)
        print("Calibración completada. Configuración guardada en "
              "'config_tablero.json'")
    except Exception as e:
        print(f"Error al guardar la configuración: {e}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()