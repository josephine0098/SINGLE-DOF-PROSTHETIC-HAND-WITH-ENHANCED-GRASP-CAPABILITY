import numpy as np

def rpm_to_rad_per_s(rpm):
    """Convert RPM to radians/second."""
    return (rpm * 2 * np.pi) / 60

def compute_required_torque(grip_force_n, lever_arm_m):
    """Compute required torque from grip force and lever arm."""
    return grip_force_n * lever_arm_m

def compute_output_rpm(motor_rpm_input, gear_ratio):
    """Calculate the output RPM after gear reduction."""
    return motor_rpm_input / gear_ratio

def compute_closing_time(required_angle_deg, output_rpm):
    """Estimate closing time based on required joint rotation and output RPM."""
    angular_speed_rad_s = rpm_to_rad_per_s(output_rpm)
    required_angle_rad = np.deg2rad(required_angle_deg)
    return required_angle_rad / angular_speed_rad_s

def evaluate_motor(grip_force_n, lever_arm_m, motor_rpm_input, gear_ratio, motor_torque_nm, max_closing_time_s, required_angle_deg=90):
    """Evaluate whether a motor meets torque and speed requirements."""
    required_torque = compute_required_torque(grip_force_n, lever_arm_m)
    output_rpm = compute_output_rpm(motor_rpm_input, gear_ratio)
    closing_time = compute_closing_time(required_angle_deg, output_rpm)

    meets_torque = motor_torque_nm >= required_torque
    meets_time = closing_time <= max_closing_time_s

    return {
        "Required Torque (Nm)": required_torque,
        "Output RPM": output_rpm,
        "Closing Time (s)": closing_time,
        "Torque OK": meets_torque,
        "Speed OK": meets_time
    }

# Example usage:
if __name__ == "__main__":
    # Inputs (example placeholder values, replace with actual when calling)
    grip_force_n = float(input("Enter required grip force (N): "))
    lever_arm_m = float(input("Enter lever arm length (m): "))
    motor_rpm_input = float(input("Enter motor free-run RPM: "))
    gear_ratio = float(input("Enter gearhead ratio: "))
    motor_torque_nm = float(input("Enter output torque of motor (Nm): "))
    max_closing_time_s = float(input("Enter max allowable closing time (s): "))

    result = evaluate_motor(
        grip_force_n,
        lever_arm_m,
        motor_rpm_input,
        gear_ratio,
        motor_torque_nm,
        max_closing_time_s
    )

    print("\nMotor Evaluation Result:")
    for key, value in result.items():
        print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
