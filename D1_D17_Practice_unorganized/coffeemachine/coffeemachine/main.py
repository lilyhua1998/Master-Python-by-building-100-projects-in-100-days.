MENU = {
    "espresso": {
        "ingredients": {
            "water": 50,
            "coffee": 18,
        },
        "cost": 1.5,
    },
    "latte": {
        "ingredients": {
            "water": 200,
            "milk": 150,
            "coffee": 24,
        },
        "cost": 2.5,
    },
    "cappuccino": {
        "ingredients": {
            "water": 250,
            "milk": 100,
            "coffee": 24,
        },
        "cost": 3.0,
    }
}

resources = {
    "water": 300,
    "milk": 200,
    "coffee": 100,
}


def coin_to_value(coffee,price):
    print(f"You need to pay ${price}.")
    total = int(input("How many quater?")) * 0.05
    total += int(input("How many quater?")) * 0.10
    total += int(input("How many quater?")) * 0.25
    total += int(input("How many quater?")) * 0.50
    return total


def paid_enough(coffee, recieved_money):
    ok = False
    while ok == False:
        if recieved_money >= MENU[coffee]["cost"]:
            return_money = round(recieved_money - MENU[coffee]["cost"], 4)
            print(f"Thanks, the extra {return_money} is giving back")
            ok = True

            return MENU[coffee]["cost"]
        else:
            left = round((MENU[coffee]["cost"] - recieved_money), 4)
            again = input(f"You still have {left} remain, do you want to pay continuously? 'y'or'n' ").lower()
            if again == 'n':
                ok = True
            else:
                recieved_money2=coin_to_value(coffee,price=left)
                recieved_money+=recieved_money2




def resource_sufficient(coffee):
    global resources
    for item in MENU[coffee]["ingredients"]:
        if MENU[coffee]["ingredients"][item] >= resources[item]:
            print(f"Sorry {item} is not enough, giving back your money.")
            return False

profit = 0
machine = "on"

while machine == "on":
    coffee = input("Please choose the below options latte/espresso/cappuccino/maintain/report").lower()

    if coffee == "maintain":
        print("Maintaing the machine. Stop process the coffee")
    elif coffee == "report":
        print(f"{resources}")
    else:
        recieved_money = coin_to_value(coffee,price=MENU[coffee]["cost"])
        profit = paid_enough(coffee, recieved_money)
        enough=resource_sufficient(coffee)
        if enough==False:
            machine = "off"
        else:
            profit += profit
            resources["profit"]=profit
            for item in MENU[coffee]["ingredients"]:
                resources[item] -= MENU[coffee]["ingredients"][item]
            print(f"Here is your {coffee}.")
            print(f"{resources}")




