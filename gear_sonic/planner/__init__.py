"""Constrained whole-body motion planning for SONIC.

Plans 29-joint trajectories with both feet pinned (OMPL ProjectedStateSpace)
and streams them to the C++ deploy over ZMQ streamed motion.
"""
