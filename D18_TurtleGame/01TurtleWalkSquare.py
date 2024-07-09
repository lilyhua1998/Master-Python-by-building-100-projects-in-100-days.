#####Turtle Intro######

import turtle as t

tim = t.Turtle()
# timmy_the_turtle.shape("turtle")
# timmy_the_turtle.color("red")
# timmy_the_turtle.forward(100)
# timmy_the_turtle.backward(200)
# timmy_the_turtle.right(90)
# timmy_the_turtle.left(180)
# timmy_the_turtle.setheading(0)


######## Challenge 1 - Draw a Square ############
tim.shape("turtle")
tim.color('black')
tim.speed(1)

for i in range(4):
    tim.forward(100)
    tim.right(90)
t.done()



