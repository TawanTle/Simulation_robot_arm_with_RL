import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
#                               SIMULATION MODE
# ---------------------------------------------------------------------------

SIM_MODE = "IK"

# Set Link (cm)
L1 = 8.0    # ความสูงฐาน
L2 = 12.0   # ท่อนแขนบน
L3 = 10.0   # ท่อนแขนล่าง

#---------------------------------------------------------------------------
#                                Forward Kinematics
#---------------------------------------------------------------------------

def forward_kinematics(theta1, theta2, theta3):
    # แปลงองศาเป็นเรเดียน
    t1 = np.radians(theta1)
    t2 = np.radians(theta2)
    t3 = np.radians(theta3)

    # ตำแหน่งฐาน
    p0 = np.array([0, 0, 0])

    # ตำแหน่งหัวไหล่
    p1 = np.array([0, 0, L1])

    # ตำแหน่งข้อศอก
    x2 = L2 * np.cos(t2) * np.cos(t1)
    y2 = L2 * np.cos(t2) * np.sin(t1)
    z2 = L1 + L2 * np.sin(t2)

    p2 = np.array([x2, y2, z2])

    # ตำแหน่งปลายแขน
    reach = L2 * np.cos(t2) + L3 * np.cos(t2 + t3)

    x3 = reach * np.cos(t1)
    y3 = reach * np.sin(t1)
    z3 = L1 + L2 * np.sin(t2) + L3 * np.sin(t2 + t3)

    p3 = np.array([x3, y3, z3])

    return p0, p1, p2, p3
# ----------------------------------------------------------
#                           Angle
# ---------------------------------------------------------
theta1 = 30    # ฐานหมุน
theta2 = 40    # หัวไหล่
theta3 = -50   # ข้อศอก

# ----------------------------------------------------------------------
#                      Draw Robot Arm with FK
# ----------------------------------------------------------------------
def plot_robot(theta1, theta2, theta3, ax=None):
    p0, p1, p2, p3 = forward_kinematics(
        theta1,
        theta2,
        theta3
    )

    points = np.array([p0, p1, p2, p3])

    # ตรวจสอบว่าเรียกโดยตรง หรือส่ง ax มาจาก Slider
    standalone = ax is None

    if standalone:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        # ลบภาพตำแหน่งเก่าก่อนวาดตำแหน่งใหม่
        ax.clear()

    # วาดแขนกล
    ax.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        "-o",
        linewidth=5,
        markersize=9,
        color="royalblue"
    )

    # วาด End effector
    ax.scatter(
        p3[0],
        p3[1],
        p3[2],
        color="red",
        s=100,
        label="End effector"
    )

    # ใส่ชื่อแต่ละตำแหน่ง
    names = ["Base", "Shoulder", "Elbow", "End"]

    for point, name in zip(points, names):
        ax.text(
            point[0],
            point[1],
            point[2] + 0.5,
            name
        )

    max_range = L1 + L2 + L3

    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(0, max_range)

    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")

    ax.set_title(
        f"3DOF Robotic Arm\n"
        f"θ1={theta1:.0f}°, "
        f"θ2={theta2:.0f}°, "
        f"θ3={theta3:.0f}°\n"
        f"End: X={p3[0]:.2f}, "
        f"Y={p3[1]:.2f}, "
        f"Z={p3[2]:.2f} cm"
    )

    ax.legend()
    ax.grid(True)

    # แสดงกราฟเองเฉพาะตอนรัน Robot_sim.py โดยตรง
    if standalone:
        plt.show()

    return p3
# -----------------------------------------------------------------------------
#                       SELECT FUNCTION MODE FK SIMULATIONS
# -----------------------------------------------------------------------------

if __name__ == "__main__" and SIM_MODE.upper() == "FK":
    theta1 = 30
    theta2 = 40
    theta3 = -50

    end_position = forward_kinematics(
        theta1,
        theta2,
        theta3
    )[3]

    print("=" * 40)
    print("Forward Kinematics")
    print("=" * 40)

    print("มุมที่กำหนด")
    print(f"Theta 1 = {theta1:.2f} องศา")
    print(f"Theta 2 = {theta2:.2f} องศา")
    print(f"Theta 3 = {theta3:.2f} องศา")

    print("\nตำแหน่งปลายแขน")
    print(f"X = {end_position[0]:.2f} cm")
    print(f"Y = {end_position[1]:.2f} cm")
    print(f"Z = {end_position[2]:.2f} cm")

    plot_robot(theta1, theta2, theta3)
# -----------------------------------------------------------------------
#                           Inverse Kinematics
# -----------------------------------------------------------------------

def inverse_kinematics(x, y, z, elbow="down"):
    # มุมหมุนของฐาน
    theta1 = np.arctan2(y, x)

    # ระยะในแนวราบจากแกนฐาน
    r = np.sqrt(x**2 + y**2)

    # ความสูงเทียบกับข้อต่อหัวไหล่
    z_relative = z - L1

    # คำนวณมุมข้อศอก
    D = (
        r**2 + z_relative**2 - L2**2 - L3**2
    ) / (2 * L2 * L3)

    if D < -1 or D > 1:
        print("ตำแหน่งนี้อยู่นอกระยะที่แขนเอื้อมถึง")
        return None

    if elbow == "up":
        theta3 = -np.arccos(D)
    else:
        theta3 = np.arccos(D)

    # คำนวณมุมหัวไหล่
    theta2 = (
        np.arctan2(z_relative, r)
        - np.arctan2(
            L3 * np.sin(theta3),
            L2 + L3 * np.cos(theta3)
        )
    )

    return (
        np.degrees(theta1),
        np.degrees(theta2),
        np.degrees(theta3)
    )

# -----------------------------------------------------------------------------
#                       SELECT FUNCTION MODE IK SIMULATIONS
# -----------------------------------------------------------------------------

target_x = 15
target_y = 5
target_z = 12

if __name__ == "__main__" and SIM_MODE.upper() == "IK":
    angles = inverse_kinematics(
        target_x,
        target_y,
        target_z,
        elbow="down"
    )

    print("=" * 40)
    print("Inverse Kinematics")
    print("=" * 40)

    print("ตำแหน่งเป้าหมาย")
    print(f"X = {target_x:.2f} cm")
    print(f"Y = {target_y:.2f} cm")
    print(f"Z = {target_z:.2f} cm")

    if angles is not None:
        theta1, theta2, theta3 = angles

        print("\nมุมที่คำนวณได้")
        print(f"Theta 1 = {theta1:.2f} องศา")
        print(f"Theta 2 = {theta2:.2f} องศา")
        print(f"Theta 3 = {theta3:.2f} องศา")

        plot_robot(theta1, theta2, theta3)