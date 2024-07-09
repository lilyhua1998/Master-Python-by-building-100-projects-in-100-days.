import csv

# temp=[]
# with open ("weather_data.csv") as weather_data:
#     data = csv.reader(weather_data)
#     for row in data:
#         # challenge 1 make a list of temp:
#         if row[1] != "temp":
#           temp.append(int(row[1]))
#
# print(temp)


# import pandas
# weather_data=pandas.read_csv("weather_data.csv")
# print(weather_data)
# #pandas can beatifually print the table format type data
#
# #pandas can automaticall recongize the first horizonal line is parameter
# temp=weather_data["temp"]
# print(temp)

###########################################
import pandas
squirrelData=pandas.read_csv("2018_Central_Park_Squirrel_Census_-_Squirrel_Data_20240709.csv")

#pandas can automaticall recongize the first horizonal line is parameter
#def findColorNum(chooseColor):
def findcolornum(chooseColor):
  rightColor=squirrelData[squirrelData["Primary Fur Color"] == chooseColor].shape[0]
  #print(rightColor)
  return rightColor


total_num=[]
chooseColors=["Gray",'Cinnamon', 'Black']
for color in chooseColors:
    num=findcolornum(color)
    total_num.append(num)
print(total_num)

color_dic={
    "Primary Ful Color": chooseColors,
    "Total Num": total_num
}

df=pandas.DataFrame(color_dic)
df.to_csv("Output/threecolor.csv")


