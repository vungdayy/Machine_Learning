"""
Ví dụ: Overfitting và khắc phục bằng K-Fold Cross Validation
Dữ liệu: chiều cao (cm) - cân nặng (kg) của 15 người
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

# ------------------------------------------------------------------
# 1. DỮ LIỆU (đúng như trong ảnh)
# ------------------------------------------------------------------
chieu_cao = np.array([147, 150, 153, 155, 158, 160, 163, 165,
                       168, 170, 173, 175, 178, 180, 183], dtype=float)
can_nang  = np.array([49, 50, 51, 52, 54, 56, 58, 59,
                       60, 72, 63, 64, 66, 67, 68], dtype=float)
# (lưu ý: điểm STT=10 (170cm - 72kg) hơi lệch so với xu hướng,
#  đây chính là "nhiễu" khiến mô hình bậc cao dễ bị overfitting)

X = chieu_cao.reshape(-1, 1)
y = can_nang

# ------------------------------------------------------------------
# 2. TẠO MÔ HÌNH BỊ OVERFITTING
#    Dùng hồi quy đa thức bậc rất cao (degree = 14) trên toàn bộ 15 điểm
# ------------------------------------------------------------------
degree_overfit = 14  # gần bằng số điểm dữ liệu (15) -> mô hình "học vẹt"

model_overfit = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=degree_overfit),
    LinearRegression()
)
model_overfit.fit(X, y)

y_pred_train = model_overfit.predict(X)
mse_train = mean_squared_error(y, y_pred_train)
print(f"[OVERFIT] Bậc đa thức = {degree_overfit}")
print(f"[OVERFIT] MSE trên chính tập huấn luyện = {mse_train:.4f}  <-- rất nhỏ, gần bằng 0")

# Vẽ đường cong của mô hình overfit
x_line = np.linspace(chieu_cao.min(), chieu_cao.max(), 300).reshape(-1, 1)
y_line_overfit = model_overfit.predict(x_line)

plt.figure(figsize=(7, 5))
plt.scatter(chieu_cao, can_nang, color="red", label="Dữ liệu thực tế", zorder=5)
plt.plot(x_line, y_line_overfit, color="blue",
         label=f"Mô hình bậc {degree_overfit} (overfitting)")
plt.title("Mô hình bị OVERFITTING (đa thức bậc 14, khớp gần như tuyệt đối)")
plt.xlabel("Chiều cao (cm)")
plt.ylabel("Cân nặng (kg)")
plt.ylim(30, 90)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("1_overfit.png", dpi=130)
plt.close()

# ------------------------------------------------------------------
# 3. DÙNG K-FOLD CROSS VALIDATION ĐỂ PHÁT HIỆN & KHẮC PHỤC
#    Thử nhiều bậc đa thức khác nhau, đánh giá bằng K-Fold (k=5)
#    -> chọn bậc cho sai số trung bình trên tập kiểm tra (validation) thấp nhất
# ------------------------------------------------------------------
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

degrees = range(1, 13)
train_mse_list = []
cv_mse_list = []

for d in degrees:
    model_d = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=d),
        LinearRegression()
    )

    # MSE trên tập train (toàn bộ) - để so sánh
    model_d.fit(X, y)
    train_mse = mean_squared_error(y, model_d.predict(X))
    train_mse_list.append(train_mse)

    # MSE trung bình qua K-Fold (đo khả năng tổng quát hoá thực sự)
    scores = cross_val_score(model_d, X, y, cv=kf,
                              scoring="neg_mean_squared_error")
    cv_mse = -scores.mean()
    cv_mse_list.append(cv_mse)

    print(f"Bậc {d:2d} | MSE train = {train_mse:8.3f} | MSE trung bình {k}-Fold = {cv_mse:10.3f}")

best_degree = list(degrees)[int(np.argmin(cv_mse_list))]
print(f"\n=> Bậc đa thức tốt nhất theo {k}-Fold CV: {best_degree} "
      f"(MSE CV nhỏ nhất = {min(cv_mse_list):.3f})")

# ------------------------------------------------------------------
# 4. VẼ SO SÁNH: train error luôn giảm, nhưng CV error tăng lại khi bậc quá cao
# ------------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(list(degrees), train_mse_list, "o-", color="green", label="MSE trên tập train")
plt.plot(list(degrees), cv_mse_list, "o-", color="orange", label=f"MSE trung bình {k}-Fold CV")
plt.axvline(best_degree, color="gray", linestyle="--",
            label=f"Bậc tối ưu = {best_degree}")
plt.yscale("log")
plt.xlabel("Bậc đa thức (độ phức tạp mô hình)")
plt.ylabel("MSE (thang log)")
plt.title("Train MSE vs K-Fold CV MSE theo bậc đa thức")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("2_train_vs_cv.png", dpi=130)
plt.close()

# ------------------------------------------------------------------
# 5. MÔ HÌNH SAU KHI KHẮC PHỤC (dùng bậc tối ưu tìm được từ K-Fold)
# ------------------------------------------------------------------
model_best = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=best_degree),
    LinearRegression()
)
model_best.fit(X, y)
y_line_best = model_best.predict(x_line)

plt.figure(figsize=(7, 5))
plt.scatter(chieu_cao, can_nang, color="red", label="Dữ liệu thực tế", zorder=5)
plt.plot(x_line, y_line_overfit, color="blue", alpha=0.4, linestyle="--",
         label=f"Mô hình cũ bậc {degree_overfit} (overfitting)")
plt.plot(x_line, y_line_best, color="darkgreen", linewidth=2.5,
         label=f"Mô hình mới bậc {best_degree} (chọn bằng {k}-Fold CV)")
plt.title("Mô hình sau khi khắc phục Overfitting bằng K-Fold Cross Validation")
plt.xlabel("Chiều cao (cm)")
plt.ylabel("Cân nặng (kg)")
plt.ylim(30, 90)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("3_after_fix.png", dpi=130)
plt.close()

print("\nĐã lưu 3 hình: 1_overfit.png, 2_train_vs_cv.png, 3_after_fix.png")