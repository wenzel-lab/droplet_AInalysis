import cv2

# Captura de video desde la cámara (0 para la cámara predeterminada)
captura = cv2.VideoCapture(0)

if not captura.isOpened():
    print("Error al abrir la cámara")
    exit()

# Nombre de la ventana
nombre_ventana = 'Cámara'

# Creación de una ventana que se reutilizará para mostrar todas las imágenes
cv2.namedWindow(nombre_ventana)

try:
    while True:
        # Lee una imagen de la cámara
        ret, imagen = captura.read()

        if not ret:
            print("Error al capturar la imagen")
            break

        # Muestra la imagen en la ventana creada
        cv2.imshow(nombre_ventana, imagen)

        # Espera 500 ms (0.5 segundos) para la siguiente captura
        # Si el usuario presiona la tecla 'q' durante este tiempo, se saldrá del bucle
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
finally:
    # Asegura la liberación del recurso de captura de la cámara y el cierre de la ventana
    captura.release()
    cv2.destroyAllWindows()
