from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from geopandas import GeoSeries


def show_plan(plan):
    ax = plt.gca()
    block_shapes = [Polygon(block["coords"]) for block in plan["blocks"]]
    building_shapes = []
    for block in plan["blocks"]:
        for building in block["buildings"]:
            building_shapes.append(Polygon(building["coords"]))
    GeoSeries(block_shapes).plot(ax=ax, color="blue")
    GeoSeries(building_shapes).plot(ax=ax, color="red")
    ax.set_aspect(1)
    plt.axis('off')
    plt.show()


def test():
    for line in open("plans.txt"):
        plan = eval(line.strip("\n"))
        print(plan)
        show_plan(plan)


if __name__ == "__main__":
    test()

