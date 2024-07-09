import colorgram

# extract the color in HirstPaint.png this image, the later number 30 is represent how nuch color yo want to extract
color = colorgram.extract('HirstPaint.jpg', 30)
# print(color)

# generate the new color list
newcolor = []


def new_color():
    for part_color in color:
        r = part_color.rgb.r
        g = part_color.rgb.g
        b = part_color.rgb.b
        rgb = (r, g, b)
        newcolor.append(rgb)
    return newcolor


print(new_color())

# new color set
color_list = [
    (61, 92, 128), (196, 163, 112), (139, 81, 62),
    (142, 163, 184), (220, 203, 126), (128, 73, 88), (198, 96, 80), (183, 181, 39), (135, 195, 152), (65, 38, 49),
    (190, 152, 168), (26, 28, 34), (163, 22, 16), (46, 114, 82), (127, 32, 43), (36, 165, 146), (185, 87, 93),
    (59, 57, 87), (163, 210, 179), (32, 94, 54), (226, 176, 168), (219, 173, 190), (119, 118, 149), (181, 189, 208),
    (59, 77, 39), (49, 69, 73)]
