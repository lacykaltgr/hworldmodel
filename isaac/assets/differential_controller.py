import torch

class DifferentialController:
    """
    This controller uses a unicycle model of a differential drive. The Controller consumes a command in the form of a linear and angular velocity, and then computes the circular arc that satisfies this command given the distance between the wheels.  This can then be used to compute the necessary angular velocities of the joints that will propell the midpoint between the wheels along the curve. The conversion is

        .. math::

            \omega_R = \\frac{1}{2r}(2V + \omega b) \n
            \omega_L = \\frac{1}{2r}(2V - \omega b)

    where :math:`\omega` is the desired angular velocity, :math:`V` is the desired linear velocity, :math:`r` is the radius of the wheels, and :math:`b` is the distance between them.


    Args:
        name (str): [description]
        wheel_radius (float): Radius of left and right wheels in cms
        wheel_base (float): Distance between left and right wheels in cms
        max_linear_speed (float): OPTIONAL: limits the maximum linear speed that will be produced by the controller. Defaults to 1E20.
        max_angular_speed (float): OPTIONAL: limits the maximum angular speed that will be produced by the controller. Defaults to 1E20.
        max_wheel_speed (float): OPTIONAL: limits the maximum wheel speed that will be produced by the controller. Defaults to 1E20.
    """

    def __init__(
        self,
        name: str,
        wheel_radius: float,
        wheel_base: float,
        max_linear_speed: float = 1.0e20,
        max_angular_speed: float = 1.0e20,
        max_wheel_speed: float = 1.0e20,
    ) -> None:
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.max_wheel_speed = max_wheel_speed

        assert self.max_linear_speed >= 0
        assert self.max_angular_speed >= 0
        assert self.max_wheel_speed >= 0

    def forward(self, command: torch.Tensor) -> torch.Tensor:
        """convert from desired [signed linear speed, signed angular speed] to [Left Drive, Right Drive] joint targets.

        Args:
            command (np.ndarray): desired vehicle [forward, rotation] speed

        Returns:
            ArticulationAction: the articulation action to be applied to the robot.
        """

        if command.shape[-1] != 2:
            raise Exception("command should have shape (batch_size, 2)")

        # limit vehicle speed
        command = torch.clamp(
            command,
            min=torch.tensor([-self.max_linear_speed, -self.max_angular_speed], device=command.device),
            max=torch.tensor([self.max_linear_speed, self.max_angular_speed], device=command.device)
        )

        # calculate wheel speed
        left_wheel_speed = ((2 * command[:, 0]) - (command[:, 1] * self.wheel_base)) / (2 * self.wheel_radius)
        right_wheel_speed = ((2 * command[:, 0]) + (command[:, 1] * self.wheel_base)) / (2 * self.wheel_radius)
        joint_velocities = torch.stack((left_wheel_speed, right_wheel_speed), dim=1)

        joint_velocities = torch.clamp(
            joint_velocities,
            min=-self.max_wheel_speed,
            max=self.max_wheel_speed
        )

        return joint_velocities
