import turtle as t

tim=t.Turtle()
TimColor=["red","blue","green","black","orange","pink","purple","yellow"]


for i in range(8):
  tim.color(TimColor[i])
  #print(i)
  line=i+3
  for _ in range (line):
      tim.forward(100)
      tim.right(360/line)

t.done()
