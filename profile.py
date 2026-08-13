"""CloudLab profile for QKD-SAGIN RL training.

This file is executed by CloudLab when the profile is created from the
Git repository. It defines one compute node, the OS image, and the setup
script that installs the Python environment and prepares the dataset.
"""

import geni.portal as portal
import geni.rspec.pg as rspec


portal.context.defineParameter(
    "hardware_type",
    "Hardware type",
    portal.ParameterType.STRING,
    "c6525-25g",
)
portal.context.defineParameter(
    "disk_image",
    "Disk image URN",
    portal.ParameterType.STRING,
    "urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU22-64-STD",
)

params = portal.context.bind()

node = rspec.Node("train")
node.hardware_type = params.hardware_type
node.disk_image = params.disk_image
node.addService(
    rspec.Execute(
        shell="bash",
        command="bash /local/repository/install.sh",
    )
)

portal.context.printRequestRSpec()
