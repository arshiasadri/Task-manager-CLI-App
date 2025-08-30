def CHATBOT():
    print("HI I am CHATBOT FOR EXIT PRINT 'exit'")
    while True:
        user_input = input("you : ?").lower()
        if user_input == "hello" :
            print("CHATBOT : hi! how are you ?")
        elif user_input == "fine":
            print("CHATBOT:i`m happy for you")
        elif user_input == "how are you":
            print("CHATBOT : im always fine because im robot!!!!!!")
        elif user_input == "exit":
            print("CHATBOT : bye")
            break
        else:
            print("I dont anderstand please try again!!!!")
CHATBOT()
