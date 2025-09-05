# %%
print(r"""
 _____                                                                    _____ 
( ___ )------------------------------------------------------------------( ___ )
 |   |                                                                    |   | 
 |   |                                                                    |   | 
 |   |    $$\      $$\                 $$$$$$$\  $$\      $$\ $$$$$$\     |   | 
 |   |    $$$\    $$$ |                $$  __$$\ $$$\    $$$ |\_$$  _|    |   | 
 |   |    $$$$\  $$$$ |$$\   $$\       $$ |  $$ |$$$$\  $$$$ |  $$ |      |   | 
 |   |    $$\$$\$$ $$ |$$ |  $$ |      $$$$$$$\ |$$\$$\$$ $$ |  $$ |      |   | 
 |   |    $$ \$$$  $$ |$$ |  $$ |      $$  __$$\ $$ \$$$  $$ |  $$ |      |   | 
 |   |    $$ |\$  /$$ |$$ |  $$ |      $$ |  $$ |$$ |\$  /$$ |  $$ |      |   | 
 |   |    $$ | \_/ $$ |\$$$$$$$ |      $$$$$$$  |$$ | \_/ $$ |$$$$$$\     |   | 
 |   |    \__|     \__| \____$$ |      \_______/ \__|     \__|\______|    |   | 
 |   |                 $$\   $$ |                                         |   | 
 |   |                 \$$$$$$  |                                         |   | 
 |   |                  \______/                                          |   | 
 |   |                                                                    |   | 
 |___|                                                                    |___| 
(_____)------------------------------------------------------------------(_____)
""")
vazn = input("please enter your weight: ")
ghad = input("please enter your height: ")
# %%
vazn = float(vazn)
ghad = float(ghad)
ghad = ghad / 100
BMI = vazn / (ghad**2)
BMI = round(BMI)
print('Your BMI is:',BMI)
# %%
if BMI < 18.5 :
    print("you are so thin!!!!")
if BMI <= 18.5 :
    print("Underweight")
    print("eat some food")
elif 18.5 <= BMI <= 25:
    print("Normal")
elif 25 <= BMI <= 30:
    print("Overweight")
elif BMI > 30:
    print("you are so fat!!!!")    
else:
    print("numbers incorrect")