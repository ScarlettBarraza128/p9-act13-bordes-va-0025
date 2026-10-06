
import cv2

# Emily Barraza NC = 0025
# Ejemplo 1 Detección de bordes con Canny
print("Ejemplo 1 Detección de bordes con Canny")

# Cargar la imagen
imagen = cv2.imread("Hipopótamo.jpg")

# Comprobar que la imagen fue cargada
if imagen is None:
    print("Error: no se pudo cargar la imagen.")
    exit()

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Detectar bordes mediante Canny
bordes = cv2.Canny(gris, 50, 150)

# Mostrar resultados
cv2.imshow("Imagen original 0025", imagen)
cv2.imshow("Imagen en escala de grises 0025", gris)
cv2.imshow("Bordes Canny 0025", bordes)

# Guardar resultado
cv2.imwrite("Bordes_Hipopótamo.jpg", bordes)

print("Detección de bordes completada.")
print("Resultado guardado: Bordes_Hipopótamo.jpg")

# Esperar una tecla
cv2.waitKey(0)

# Cerrar ventanas
cv2.destroyAllWindows()
print("Emily Barraza NC = 0025")


# Ejemplo 4 Identificación de formas geométricas
print("Ejemplo 4 Identificación de formas geométricas")

# Cargar imagen
imagen = cv2.imread("Hipopótamo.jpg")

# Comprobar imagen
if imagen is None:
    print("Error: no se pudo cargar la imagen.")
    exit()

# Escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Umbralización
_, binaria = cv2.threshold(
    gris,
    127,
    255,
    cv2.THRESH_BINARY
)

# Buscar contornos
contornos, _ = cv2.findContours(
    binaria,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Crear copia
resultado = imagen.copy()

for contorno in contornos:

    # Eliminar objetos pequeños
    area = cv2.contourArea(contorno)

    if area < 500:
        continue

    # Perímetro
    perimetro = cv2.arcLength(
        contorno,
        True
    )

    # Aproximar contorno
    aproximacion = cv2.approxPolyDP(
        contorno,
        0.04 * perimetro,
        True
    )

    # Número de vértices
    vertices = len(aproximacion)

    # Identificar forma
    if vertices == 3:
        forma = "Triangulo"

    elif vertices == 4:

        x, y, ancho, alto = cv2.boundingRect(
            aproximacion
        )

        relacion = ancho / float(alto)

        if 0.90 <= relacion <= 1.10:
            forma = "Cuadrado"
        else:
            forma = "Rectangulo"

    elif vertices == 5:
        forma = "Pentagono"

    elif vertices > 5:
        forma = "Circulo"

    else:
        forma = "Desconocida"

    # Dibujar contorno
    cv2.drawContours(
        resultado,
        [aproximacion],
        -1,
        (0, 255, 0),
        2
    )

    # Obtener posición
    x, y, ancho, alto = cv2.boundingRect(
        aproximacion
    )

    # Escribir nombre de la forma
    cv2.putText(
        resultado,
        forma,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

# Mostrar resultado
cv2.imshow(
    "Formas identificadas",
    resultado
)

# Guardar resultado
cv2.imwrite(
    "Formas_Hipopótamo.jpg",
    resultado
)

print("Identificación de formas terminada.")
print("Resultado guardado: Formas_Hipopótamo.jpg")

# Esperar
cv2.waitKey(0)

# Cerrar
cv2.destroyAllWindows()

print("Emily Barraza NC = 0025")
