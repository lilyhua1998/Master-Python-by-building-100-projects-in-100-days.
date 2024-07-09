from datetime import date
today=date.today()
today=today.strftime('%B %d, %Y')
print(f"Today is {today} ")

print('The daily task is below')

again=True
sum=0
sum_weekly=0
list=[]
interview=0
chance={}
while again==True:
    task=int(input("how many job did you apply today?"))
    sum += task
    sum_weekly += task
    print(f"now you apply total {sum} job.")
    list.append(task)
    if len(list)%7==0:
        print(f"This week we have {sum_weekly} job applied.")
        print(f"from Mon. to Sun., you apply {list[-7:]} job in each day.")
        sum_weekly=0

        interview_get=input("Do you get any interview? y or n").lower()
        if interview_get=='y':
            num=int(input("How many interview do you get this week?"))
            interview+=num

            for i in range(0,num):
                company=input("Company name?")
                job_title=input("Job title?")
                salary=input("salary?")
                chance[company]=[job_title, salary]
            print(chance)

        job=input("Did you find the job? y or n").lower()
        if job =="y":
            again=False
            print(f"You have applied {sum} job this cycle.")

