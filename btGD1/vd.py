import numpy as np
import matplotlib.pyplot as plt

# 1. Khởi tạo dữ liệu giả lập (tương tự đồ thị bên trái)
np.random.seed(42)
N = 1000
X = np.random.rand(N, 1) # X trong khoảng [0, 1]
# Giả sử hàm thực tế là y = 4 + 3x + nhiễu (noise)
y = 4 + 3 * X + np.random.randn(N, 1) * 0.2

# Thêm bias (cột toàn số 1) vào ma trận X để tính toán w0
X_bar = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

# 2. Định nghĩa hàm mất mát và đạo hàm
def cost(w):
    # Hàm mất mát MSE: L(w) = 1/(2N) * ||X*w - y||^2
    return 0.5/N * np.linalg.norm(X_bar.dot(w) - y, 2)**2

def grad(w):
    # Đạo hàm: grad = 1/N * X^T * (X*w - y)
    return 1/N * X_bar.T.dot(X_bar.dot(w) - y)

# Thuật toán Gradient Descent
def myGD(w_init, eta):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] - eta * grad(w[-1])
        if np.linalg.norm(grad(w_new)) < 1e-4:
            break
        w.append(w_new)
    return (w, it)

# 3. Chạy thuật toán
w_init = np.array([[2.0], [1.0]]) # Điểm khởi tạo như trong hình phải
eta = 0.5 # Tốc độ học (learning rate)
w_hist, iters = myGD(w_init, eta)
w_hist_np = np.array(w_hist).reshape(-1, 2)

# 4. Vẽ biểu đồ
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- BIỂU ĐỒ TRÁI: Dữ liệu và các đường hồi quy qua từng vòng lặp ---
ax1.plot(X, y, 'b.', markersize=4, label='Data points') # Điểm dữ liệu màu xanh
x0 = np.linspace(0, 1, 2, endpoint=True)

# Vẽ một số đường trung gian (màu đỏ mờ)
step = max(1, len(w_hist)//15)
for i in range(0, len(w_hist), step):
    w_tmp = w_hist[i]
    y0 = w_tmp[0][0] + w_tmp[1][0]*x0
    ax1.plot(x0, y0, 'r-', alpha=0.3)

# Vẽ đường hồi quy cuối cùng (đậm)
w_final = w_hist[-1]
y_final = w_final[0][0] + w_final[1][0]*x0
ax1.plot(x0, y_final, 'r-', linewidth=2.5, label='Final Regression Line')
ax1.set_xlim(0.0, 1.0)
ax1.set_ylim(0, 10)
ax1.set_title("Sự thay đổi của đường hồi quy")
ax1.legend()

# --- BIỂU ĐỒ PHẢI: Đường đồng mức (Contour) của hàm mất mát ---
# Tạo lưới giá trị cho w0 và w1
w0_vals = np.linspace(1.5, 5.5, 100)
w1_vals = np.linspace(0.5, 4.0, 100)
W0, W1 = np.meshgrid(w0_vals, w1_vals)
Z = np.zeros_like(W0)

# Tính giá trị cost cho từng điểm trên lưới
for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        Z[i, j] = cost(np.array([[W0[i, j]], [W1[i, j]]]))

# Vẽ nền màu và đường đồng mức
contour_filled = ax2.contourf(W0, W1, Z, 50, cmap='jet', alpha=0.6)
ax2.contour(W0, W1, Z, 20, colors='darkblue', linewidths=0.5)

# Vẽ quỹ đạo của Gradient Descent (các điểm màu đỏ nối với nhau)
ax2.plot(w_hist_np[:, 0], w_hist_np[:, 1], 'ro-', markersize=5, linewidth=1.5)
ax2.set_xlim(1.5, 5.5)
ax2.set_ylim(0.5, 4.0)
ax2.set_title("Quỹ đạo Gradient Descent trên hàm mất mát")
ax2.set_xlabel("w0 (Intercept)")
ax2.set_ylabel("w1 (Slope)")

plt.suptitle("Tối ưu hàm mất mát của Linear Regression bằng GD", fontsize=16)
plt.tight_layout()
plt.show()