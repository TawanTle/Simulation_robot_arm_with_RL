import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from Simulation_robot_arm_with_RL.Robot_sim import plot_robot

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# เว้นพื้นที่ใต้กราฟ
plt.subplots_adjust(bottom=0.25)

# วาดโดยใช้ฟังก์ชันจาก Robot_sim.py
plot_robot(0, 30, -30, ax=ax)

# สร้างพื้นที่ Slider
ax_base = plt.axes([0.20, 0.15, 0.65, 0.03])
ax_shoulder = plt.axes([0.20, 0.10, 0.65, 0.03])
ax_elbow = plt.axes([0.20, 0.05, 0.65, 0.03])

base_slider = Slider(
    ax_base, "Base",
    -90, 90,
    valinit=0,
    valstep=1
)

shoulder_slider = Slider(
    ax_shoulder, "Shoulder",
    -20, 90,
    valinit=30,
    valstep=1
)

elbow_slider = Slider(
    ax_elbow, "Elbow",
    -120, 120,
    valinit=-30,
    valstep=1
)

def update(value):
    # บันทึกมุมกล้องเอาไว้
    elevation = ax.elev
    azimuth = ax.azim

    # ฟังก์ชันนี้จะลบภาพเก่าและวาดใหม่ให้เอง
    plot_robot(
        base_slider.val,
        shoulder_slider.val,
        elbow_slider.val,
        ax=ax
    )

    # คืนมุมกล้องเดิม
    ax.view_init(elev=elevation, azim=azimuth)
    fig.canvas.draw_idle()

base_slider.on_changed(update)
shoulder_slider.on_changed(update)
elbow_slider.on_changed(update)

plt.show()