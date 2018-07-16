import shapely
from shapely.geometry import Polygon, Point

a = Polygon([[0, 0], [0, 1], [1, 2], [2, 0]])
print(a.bounds)
print(a.boundary.bounds)
print(a[0])