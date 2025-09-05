#             حچم و مساحت استوانه
pi = 22/7
height = float(input("Height of cylinder: "))
radian = float(input("radius of cylinder: "))
volume = pi * radian * radian * height
sur_area = ((2 * pi * radian)* height) + ((pi * radian ** 2) * 2)
print("Volume is: ", volume)
print("surface Area is: ", sur_area)