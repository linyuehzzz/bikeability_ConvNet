def extract_first_three_lanes(polys):
    """
    Extract the first three lanes

    Input
      polys:   all lanes

    Ouput
      the first three lanes
    """
    return polys[:3]


def calculate_max_width(poly):
    """
    Calculate the maximum width of a polygon and
    the cooresponding y coordinate of the vertex used to calculate the maximum width

    Input
      poly:   a set of vertices of a polygon

    Ouput
      width_y: the y coordinate of the vertex used to calculate the maximum width
    """
    width = 0
    width_y = 0
    for p0 in poly:
        x0, y0 = p0[0], p0[1]
        for i in range(len(poly)):
            x1, y1 = poly[i-1][0], poly[i-1][1]
            x2, y2 = poly[i][0], poly[i][1]
            if y0 == y1 == y2:
                if abs(x1 - x2) > width:
                    width = abs(x1 - x2)
                    width_y = y0
            elif y0 != y1 != y2:
                x = (y0 - y2)/(y1 - y2) * (x1 - x2) + x2
                if x > x0 and x - x0 > width:
                    width = x - x0
                    width_y = y0
    return width_y


def calculate_max_y(width_ys):
    """
    Calculate the y coordinate of the baseline used for width comparisons

    Input
      width_ys:   a collection of y coordinates used to calculate the maximum widths

    Ouput
      the y coordinate of the baseline
    """
    return max(width_ys)

def calculate_compared_width(y_base, poly):
    """
    Calculate the width of each polygon according to the baseline

    Input
      y_base:   y coordinate of the base line
      poly: a set of vertices of a polygon

    Ouput
      width: the width of a polygon
    """
    width = 0
    width_xs = []
    for i in range(len(poly)):
        x1, y1 = poly[i - 1][0], poly[i - 1][1]
        x2, y2 = poly[i][0], poly[i][1]
        if y_base == y1 == y2:
            if abs(x1 - x2) > width:
                width = abs(x1 - x2)
        elif y_base != y1 != y2:
            x = (y_base - y2) / (y1 - y2) * (x1 - x2) + x2
            width_xs.append(x)
    if max(width_xs) - min(width_xs) > width:
        width = max(width_xs) - min(width_xs)
    return width

def compare_widths(polys):
    """
    Calculate the index of the polygon with the maximum width

    Input
      polys: a set of polygons

    Ouput
      index and the corresponding width
    """
    # 1. Extract the first three lanes
    polys = extract_first_three_lanes(polys)

    # 2. Calculate the y coordinate of the copmared baseline
    width_ys = []
    for poly in polys:
        width_y = calculate_max_width(poly)
        width_ys.append(width_y)
    y_base = calculate_max_y(width_ys)

    # 3. Compare widths
    width = 0
    i = 0
    for poly in polys:
        w = calculate_compared_width(y_base, poly)
        if w > width:
            width = w
            i = polys.index(poly)

    indexes = [idx for idx in range(3) if idx != i]
    polys_results = [poly for poly in polys if polys.index(poly) != i]
    return indexes, polys_results


polygon1 = [[0,0], [4,0], [3,9], [1.5,9], [0,0]]
polygon2 = [[1.5,0], [6,0], [8,8], [4,8], [1.5,0]]
polygon3 = [[4,0], [8,0], [5,7], [4,7], [4,0]]
polygon4 = [[1.5,0], [6,0], [8,8], [4,8], [1.5,0]]
polygons = [polygon1, polygon2, polygon3, polygon4]
print(polygons)
print(compare_widths(polygons))