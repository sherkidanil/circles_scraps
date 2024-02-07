from typing import Tuple, List

import cv2
import numpy as np
import matplotlib.pyplot as plt

def circle_intersection(circle1: Tuple[int], circle2: Tuple[int]) -> List[Tuple[float]]:
    """Находит пересечения двух кругов.

    Args:
        circle1 (Tuple[int]): кортеж с тремя параметрами первой окружности - x центра, y центра, радиус.
        circle2 (Tuple[int]): кортеж с тремя параметрами второй окружности - x центра, y центра, радиус.

    Returns:
        List[Tuple[float]]: список из двух точек в формате кортежа.
    """
    x0, y0, r0 = circle1
    x1, y1, r1 = circle2

    # расстояние между центрами
    d = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    
    # проверка на то, чтобы точек пересечения было 2
    if d > r1 + r0 or d < abs(r1 - r0) or d == 0 and r1 == r0:
        return None
    
    # отрезки
    a = (r0**2 - r1**2 + d**2) / (2 * d)
    h = np.sqrt(r0**2 - a**2)

    # вспомогательная точка
    x2 = x0 + a / d * (x1 - x0) 
    y2 = y0 + a / d * (y1 - y0) 
    
    x3 = x2 + h / d * (y1 - y0) 
    y3 = y2 - h / d * (x1 - x0) 
    x4 = x2 - h / d * (y1 - y0) 
    y4 = y2 + h / d * (x1 - x0) 
    
    return [(x3, y3), (x4, y4)]

def find_scrap_coords(file: str,
                      dp: int = 1,
                      minDist: int = 400,
                      params: Tuple[int] = (5, 10),
                      radius_limits: Tuple[int] = (0, 0),
                      visualize: bool = False) -> List[Tuple[float]]:
    """Находит круги на изображении и возвращает координаты центра и радиус каждого круга.

    Args:
        file (str): путь к файлу в формате .png.
        dp (int, optional): параметр разрешения метода Хафа. По умолчанию 1.
        minDist (int, optional): минимальное расстояние между центрами обнаруженных кругов. По умолчанию 400.
        params (Tuple[int], optional): параметры метода обнаружения кругов. По умолчанию (0, 0).
        radius_limits (Tuple[int], optional): минимальный и максимальный радиус обнаруживаемых кругов. По умолчанию (0, 0).
        visualize (bool, optional): если True, будет отображать изображение с обнаруженными кругами. По умолчанию False.

    Returns:
        List[Tuple[float]]: список координат.
    """
    
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file does not exists"
    
    circles = cv2.HoughCircles(image = img,
                               method = cv2.HOUGH_GRADIENT,
                               dp = dp,
                               minDist= minDist,
                               param1 = params[0],
                               param2 = params[1],
                               minRadius=radius_limits[0],
                               maxRadius=radius_limits[1])

    circles = np.array(circles, dtype = int)
    base_circle = tuple(circles[0, 0])
    result = []

    if visualize:
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i in circles[0,:]:
            # рисуем круг зеленым
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # отмечаем центр круга красным
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
            print(f'x center= {i[0]}, y center = {i[1]}, radius = {i[2]}')
        plt.figure(figsize = (6,6))
        plt.imshow(cimg)

    
    for i in range(1, len(circles[0,:])):
        circle = tuple(circles[0, i])
        points = circle_intersection(base_circle, circle)

        # если двух пересечений нет
        if points is None:
            next

        result.append(points[0])
        result.append(points[1])
        
        if visualize:
            for point in points:
                plt.scatter(point[0], point[1], c = 'r')
    
    if visualize:
        plt.show()

    return result
