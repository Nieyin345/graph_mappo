#!/usr/bin/env python3
# geni-lib profile for a single qkd_rl training node.
#
# This is the code front-end to the same thing the Topology Editor builds. It is
# worth using over the GUI for three things the editor cannot express:
#
#   1. A LIST of acceptable hardware types. The allocator picks whichever is
#      free, which matters a lot when a good type shows "Free 1" and you would
#      otherwise fail to instantiate.
#   2. An explicit disk_image URN -- the image baked on 2026-09-14 that already
#      carries /opt/qkd (venv + dataset + BC checkpoint).
#   3. routable_control_ip, so ssh works without any extra setup.
#
# IMPORTANT: the hardware type list the Editor shows is filtered by the CLUSTER
# selected for the experiment. A profile cannot grant access to hardware the
# project is not authorised for -- declaring "r750" on an Apt-only project just
# fails to instantiate. Change the cluster first, then the type list changes.
#
# Paste this into the profile editor at https://www.emulab.net (Experiments ->
# Create Experiment -> choose "Python" rather than the Topology Editor), or keep
# it in the repo for reference.
#
# KEY NUMBERS, measured on this workload -- read before picking hardware:
#   * Single-run speed: the laptop's RTX 4060 beats every CPU-only node tested
#     (40 s/update vs 85-129 s). The rollout is Python-bound and single-threaded,
#     so a server's slower cores cost more than its extra cores gain.
#   * The update phase is 40-60% of an iteration and 75% of THAT is the backward
#     pass, which is ~4x faster on GPU than CPU. But the GPU must be newer than
#     an RTX 4060 to be worth it; the GPUs in P100 (2016) / K80 (2014) boxes are not.
#   * Therefore: the only reason to use a server is to run SEVERAL arms at once.
#     Rank candidates by total threads x RAM, not by clock speed or GPU.

import geni.portal as portal
import geni.rspec.pg as pg

# The image baked 2026-09-14 from the Apt c6220 node. It carries /opt/qkd:
#   /opt/qkd/venv                                   python 3.10 + torch 2.14.0+cpu
#   /opt/qkd/graph_mappo/dataset/global/link_data.h5   374 MB, the only training input
#   /opt/qkd/graph_mappo/dataset/global/rate_stats.json p99 reference; WITHOUT IT the
#                                                       rate features silently use
#                                                       p99 = 10.0 instead of 12,655
#   /opt/qkd/graph_mappo/outputs/supervised_pg_phased/  BC warm start
# It lives in the Apt cluster's image store, so it can only boot Apt nodes.
IMAGE = "urn:publicid:IDN+apt.emulab.net+image+rdmaopt-PG0:c4130"

pc = portal.Context()

# List several types in preference order. The allocator takes the first one with
# a free node, so this is how you survive a type showing "Free 1".
#
# NOTE ON IMAGE PORTABILITY: I could not read the CloudLab manual to settle
# whether an image snapshotted on one node type boots on another -- docs.cloudlab.us
# is blocked from here. What is known: images are block-level Frisbee snapshots
# that record their provenance (base image + differing blocks), every node type
# has a default image, and compatibility is at least architecture-based (the
# Emulab tree carries a patch forcing an arch into imported image metadata
# because Utah mixes arm64 and x86_64 nodes). So the old "an image is welded to
# its node type" assumption is too strong. Just TRY it: instantiating with an
# incompatible image costs nothing, the portal refuses and the experiment is
# simply not created.
pc.defineParameter(
    "hardware",
    "Hardware type (first free one wins)",
    portal.ParameterType.STRING,
    "c6220",
    [
        ("c6220", "Apt        -- 32 threads /  64 GB  (the image was built here)"),
        ("r320", "Apt        -- 16 threads /  16 GB  (strictly worse, avoid)"),
        ("c240g5", "CloudLab Wisc -- 40 threads / 192 GB  + P100 GPU"),
        ("c220g5", "CloudLab Wisc -- 40 threads / 192 GB"),
    ],
)
pc.defineParameter(
    "disk_image",
    "Disk image URN",
    portal.ParameterType.STRING,
    IMAGE,
)
params = pc.bindParameters()

rspec = pg.Request()

node = pg.RawPC("train")
node.hardware_type = params.hardware
node.disk_image = params.disk_image

# A routable control IP is what makes `ssh <node>` work directly; without it the
# node is only reachable from the portal's console.
node.routable_control_ip = True

iface = node.addInterface("if0")
iface.addAddress(pg.IPv4Address("192.168.1.1", "255.255.255.0"))
rspec.addResource(node)

pc.printRequestRSpec(rspec)
