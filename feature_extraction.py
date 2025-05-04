import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import convolve
from sklearn.neighbors import NearestNeighbors
import cv2


def read_image_gray(path):
    """
    Verilen dosya yolundan görüntüyü oku ve gri tonlamaya çevir.
    """
    image = Image.open(path).convert('L')  # 'L' -> grayscale mode
    return np.array(image)

def show_image(image, title="Görüntü"):
    """
    Görüntüyü ekranda göster.
    """
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def harris_response(image, k=0.0025, window_size=3):
    """
    Harris köşe tepkisi (corner response) hesapla.
    """
    image = image.astype(np.float32)  # Görüntüyü float32'e çevir

    Ix, Iy = compute_gradients(image)

    # Ix², Iy² ve IxIy
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Küçük bir gaussian pencerede ortalamalarını al
    from scipy.ndimage import gaussian_filter
    Sxx = gaussian_filter(Ixx, sigma=1)
    Syy = gaussian_filter(Iyy, sigma=1)
    Sxy = gaussian_filter(Ixy, sigma=1)

    # Harris matrix M ve R değeri
    detM = (Sxx * Syy) - (Sxy ** 2)
    traceM = Sxx + Syy
    R = detM - k * (traceM ** 2)

    return R


def compute_gradients(image):
    """
    Görüntü üzerinde x ve y yönlerinde gradyan hesapla.
    """
    # Sobel kernel tanımı
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])

    # Gradyanları hesapla
    grad_x = convolve(image, sobel_x)
    grad_y = convolve(image, sobel_y)

    # Gradyan büyüklüğü ve yönü
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)  # Radyan cinsinden yön

    return magnitude, direction


def detect_harris_keypoints(R, threshold):
    """
    Harris response haritasından threshold ile keypoint seçimi.
    """
    max_R = np.max(R)
    keypoints = np.argwhere(R > threshold * max_R)
    keypoints = [tuple(pt) for pt in keypoints]
    return keypoints


def detect_keypoints_topk(magnitude, top_k):
    """
    Gradyan büyüklüğüne göre en büyük top_k kadar anahtar nokta seçimi.
    """
    flat_indices = np.argsort(magnitude.flatten())[::-1]  # Büyükten küçüğe sırala
    top_indices = flat_indices[:top_k]

    y_coords, x_coords = np.unravel_index(top_indices, magnitude.shape)
    keypoints = list(zip(y_coords, x_coords))
    return keypoints


    return keypoints
def compute_descriptors(image, keypoints, window_size=8):
    """
    Her keypoint çevresindeki pencereyi alarak descriptor üret.
    """
    descriptors = []
    half_window = window_size // 2
    h, w = image.shape

    for y, x in keypoints:
        if y - half_window < 0 or y + half_window >= h or x - half_window < 0 or x + half_window >= w:
            continue  # Sınırdaki noktaları atla

        window = image[y-half_window:y+half_window, x-half_window:x+half_window]
        descriptor = window.flatten()  # 8x8 -> 64 elemanlı vektör
        descriptors.append(descriptor)

    return np.array(descriptors)

def draw_keypoints(image, keypoints):
    """
    Anahtar noktaları hızlı bir şekilde görüntü üzerine çiz.
    """
    # Görüntüyü renkliye çevir (eğer griyse)
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()

    # Keypoint'leri toplu olarak çiz
    for y, x in keypoints:
        cv2.circle(vis_image, (int(x), int(y)), 2, (0, 0, 255), -1)

    # OpenCV kullanarak hızlıca göster
    filtered_image = Image.fromarray(vis_image)
    return filtered_image



def match_features(desc1, desc2, threshold):
    """
    Nearest Neighbors ile hızlı feature matching.
    """
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(desc2)
    distances, indices = nn.kneighbors(desc1)

    matches = []
    for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
        if dist < threshold:
            matches.append((i, idx))
    return matches

def draw_matches(img1, kp1, img2, kp2, matches, window_name="Feature Matches"):
    """
    İki görüntü arasında eşleşen noktaları hızlı şekilde çiz.
    """
    # İki gri görüntüyü renkliye çevir
    if len(img1.shape) == 2:
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_color = img1.copy()

    if len(img2.shape) == 2:
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2_color = img2.copy()

    # Görüntüleri yatay olarak birleştir
    canvas = np.hstack((img1_color, img2_color))

    # Eşleşmeleri hızlı şekilde çiz
    for idx1, idx2 in matches:
        y1, x1 = kp1[idx1]
        y2, x2 = kp2[idx2]

        pt1 = (int(x1), int(y1))
        pt2 = (int(x2 + img1.shape[1]), int(y2))

        # Çizgiler
        cv2.line(canvas, pt1, pt2, (0, 0, 255), 1)

        # Noktalar
        cv2.circle(canvas, pt1, 3, (0, 255, 255), -1)
        cv2.circle(canvas, pt2, 3, (0, 255, 255), -1)

    # Sonuç görüntüyü hızlıca göster
    cv2.imshow(window_name, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # Görüntüleri oku
    img1 = read_image_gray('foto1.jpg')
    img2 = read_image_gray('foto2.jpg')

    # Gradyan hesapla
    mag1, dir1 = compute_gradients(img1)
    mag2, dir2 = compute_gradients(img2)

    # # Anahtar noktaları bul
    # kp1 = detect_keypoints_topk(mag1, top_k=50)
    # kp2 = detect_keypoints_topk(mag2, top_k=50)

    # Harris köşe tepkisini hesapla
    R1 = harris_response(img1)
    R2 = harris_response(img2)

    # Anahtar noktaları bul
    kp1 = detect_harris_keypoints(R1, threshold=0.001)
    kp2 = detect_harris_keypoints(R2, threshold=0.001)


    # Descriptor çıkar
    desc1 = compute_descriptors(img1, kp1)
    desc2 = compute_descriptors(img2, kp2)

    # Anahtar noktaları çiz
    draw_keypoints(img1, kp1)
    draw_keypoints(img2, kp2)

    # Feature matching
    matches = match_features(desc1, desc2, threshold=5000)  # threshold'u duruma göre ayarlayabiliriz

    print(f"Eşleşen feature sayısı: {len(matches)}")

    # Eşleşmeleri görselleştir
    draw_matches(img1, kp1, img2, kp2, matches)


if __name__ == "__main__":
    main()