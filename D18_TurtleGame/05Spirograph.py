import turtle as t
import random

tim=t.Turtle()
#set the speed to the fatest (0-10), 0 is the fast
tim.speed(0)
#set the color to 255 RGB color mode
t.colormode(255)

#generate random color using tuple(r,g,b)
def random_color():
    r=random.randint(0,255)
    g=random.randint(0,255)
    b=random.randint(0,255)
    color=(r,g,b)
    return color

def draw_spirograph(gap):
    for _ in range (int(360/gap)): #when 360/gap(divide) will automatically turn to float, range can only unput integer
        tim.color(random_color())
        tim.circle(100) #draw the circle(radius)
        current_heading=tim.heading()
        tim.setheading(current_heading+gap)


draw_spirograph(5)
screen=t.Screen()
screen.exitonclick()