def conpound(init,interest,target):
    s = init
    month = 0
    ic =  round((init*interest)/100,2)
    n = lambda amount : amount + ic
    formt = lambda number : "{:,}".format(number)
    mtoyears = formt(target//12)
    while month < target:
        t = round(n(init),2)
        month+=1
        init = t
    print(f"At a rate of {interest}% per month,\n{s}$ will turn into {formt(t)}$ in {mtoyears} years")

conpound(7,0.97,24) # 100.000 years lmao

def co(initial,interest,time):
    T = initial**(interest*time)
    print(T)


co(initial=7,interest=0.97,time=24)