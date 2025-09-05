#              Commodity inflation rate
p = int(input("Enter arzesh kala:"))
n = int(input("Enter tedad sal haye pishbibi:"))
inc = int(input("nerkh tavarom:"))
print("Year     price")
for i in range(1, n+1):
     p = p + (p * inc / 100)
     print(i,"     ", p)