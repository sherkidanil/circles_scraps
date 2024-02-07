from algorithm import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Находит круги на изображении и возвращает координаты центра и радиус каждого круга')
    parser.add_argument('file', type=str, help='путь к файлу в формате .png')
    parser.add_argument('--dp', type=int, default=1, help='параметр разрешения метода Хафа')
    parser.add_argument('--minDist', type=int, default=400, help='минимальное расстояние между центрами обнаруженных кругов')
    parser.add_argument('--params', nargs=2, type=int, dest='params', metavar=('param1', 'param2'), default=(5, 10),
                        help='параметры метода обнаружения кругов')
    parser.add_argument('--radius_limits', nargs=2, type=int, dest='radius_limits', metavar=('minRadius', 'maxRadius'),
                        default=(0, 0), help='минимальный и максимальный радиус обнаруживаемых кругов')
    parser.add_argument('--visualize', dest='visualize', action='store_true', help='визуализация')

    args = parser.parse_args()
    result = find_scrap_coords(file = args.file, dp = args.dp, minDist =  args.minDist,
                               params = args.params,
                               radius_limits=args.radius_limits,
                               visualize = args.visualize)
    for res in result:
        print(res)
