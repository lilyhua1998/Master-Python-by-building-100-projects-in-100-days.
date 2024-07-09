import turtle as t
import random

tim=t.Turtle()
TimColor=["red","blue","green","black","orange","pink","purple","yellow"]
direction=[0, 90, 180, 270]
tim.pensize(10)
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


#the path will go 200 times and random color + random 90,180,270 angle
for _ in range(200):
    tim.color(random_color())
    tim.forward(30)
    tim.setheading(random.choice(direction))