#TODO: Create a letter using starting_letter.txt 
#for each name in invited_names.txt
#Replace the [name] placeholder with the actual name.
#Save the letters in the folder "ReadyToSend".
    
#Hint1: This method will help you: https://www.w3schools.com/python/ref_file_readlines.asp
    #Hint2: This method will also help you: https://www.w3schools.com/python/ref_string_replace.asp
        #Hint3: THis method will help you: https://www.w3schools.com/python/ref_string_strip.asp



with open ("./Input/Letters/starting_letter.txt") as letter:
    ReadLetter=letter.read()
    #print(ReadLetter)
    #Dear [name],
    # You are invited to my birthday this Saturday.
    # Hope you can make it!
    # Angela

with open("./Input/Names/invited_names.txt") as names:
    invited_names=names.readlines()
    # print(invited_names) #['Aang\n', 'Zuko\n', 'Appa\n', 'Katara\n', 'Sokka\n', 'Momo\n', 'Uncle Iroh\n', 'Toph']

for name in invited_names:
    newname=name.strip() #strip the blank part
    #print(name)
    CustomedReadLetter=ReadLetter.replace("[name]",newname) #replace [name] to invited_name
    #output the letter file
    with open(f"./Output/ReadyToSend/{newname}", mode="w") as OutputLetter: #change the default mode from read to write
        #write the letter
        OutputLetter.write(CustomedReadLetter)