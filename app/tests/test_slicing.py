import cv2

img = cv2.imread('app/assets/modelo.png')
if img is None:
    print('img e nulo')
    raise TypeError('img nao pode ser none')

x_i, y_i, x_f, y_f = [718.39, 464.99, 871.03, 637.62]

recorte = img[int(y_i) : int(y_f), int(x_i) : int(x_f)].copy()

print(f'imagem original: {img.shape}')
print(f'imagem recortada: {recorte.shape}')

cv2.imwrite('recorte.png', recorte)

cv2.imshow('recorte', recorte)
cv2.imshow('original', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
