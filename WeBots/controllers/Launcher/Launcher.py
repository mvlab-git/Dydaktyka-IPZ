"""Launcher controller."""
from controller import Robot,Motor

# Create Robot instance
robot = Robot()
launcher:Motor
# get linear motor called "Launcher" from robot
launcher = robot.getDevice("Launcher")

# Get file timeStep
timestep = int(robot.getBasicTimeStep())

# Main loop:
while robot.step(timestep) != -1:
    # Set linear motor displacement to selected position
    launcher.setPosition(0.05)

