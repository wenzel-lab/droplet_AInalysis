import cv2

# Captura de video desde la cámara (0 para la cámara predeterminada)
captura = cv2.VideoCapture(0)

if not captura.isOpened():
    print("Error al abrir la cámara")
    exit()

while True:
    # Lee una imagen de la cámara
    ret, imagen = captura.read()

    if not ret:
        print("Error al capturar la imagen")
        break

    # Muestra la imagen en una ventana llamada 'Cámara'
    cv2.imshow('Cámara', imagen)

    # Espera 500 ms (0.5 segundos)
    # Si el usuario presiona la tecla 'q' durante este tiempo, saldrá del bucle
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Libera el recurso de captura de la cámara y cierra todas las ventanas
captura.release()
cv2.destroyAllWindows()
