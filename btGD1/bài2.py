def ff(x) :
    return x**2 -1
def f(x) :
    return (1/3) * x**3 -x
def myGD1(x0,eta):
    x=[x0]
    for i in range(100) :
        x_new = x[-1]-eta*ff(x[-1])
        if abs(x_new) < 1e-10 :
            break
        x.append(x_new)
    return (x,i)
(x0,i) = myGD1(4,.1)
print(f"Giá trị x tại điểm cực tiểu: {x0[-1]:.5f}")
print(f"Giá trị cực tiểu của hàm số f(x): {f(x0[-1]):.5f}")
print(f"Số vòng lặp cần thiết: {i}")

