import os

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base
import numpy as np

_XML_PATH = os.path.join(os.path.dirname(__file__), "kamigami.xml")


class KamigamiRobot(base.Walker):
    """A Kamigami walker."""

    def _build(self, name="kamigami"):
        """Build a Kamigami walker.

        Args:
          name: name of the walker.
        """
        self._appendages_sensors = []
        self._bodies_pos_sensors = []
        self._bodies_quats_sensors = []
        self._mjcf_root = mjcf.from_path(_XML_PATH)
        if name:
            self._mjcf_root.model = name

        # Initialize previous action.
        self._prev_action = np.zeros(
            shape=self.action_spec.shape, dtype=self.action_spec.dtype
        )

    def initialize_episode(self, physics, random_state):
        self._prev_action = np.zeros_like(self._prev_action)

    def apply_action(self, physics, action, random_state):
        super().apply_action(physics, action, random_state)

        # Updates previous action.
        self._prev_action[:] = action

    def _build_observables(self):
        return KamigamiObservables(self)

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def upright_pose(self):
        return base.WalkerPose()

    @composer.cached_property
    def actuators(self):
        return self._mjcf_root.find_all("actuator")

    @composer.cached_property
    def root_body(self):
        return self._mjcf_root.find("body", "torso")

    @composer.cached_property
    def bodies(self):
        return tuple(self.mjcf_model.find_all("body"))

    @composer.cached_property
    def mocap_tracking_bodies(self):
        """Collection of bodies for mocap tracking."""
        return tuple(self.mjcf_model.find_all("body"))

    @property
    def mocap_joints(self):
        return self.mjcf_model.find_all("joint")

    @property
    def _leg_bodies(self):
        return (
            self._mjcf_root.find("body", "left_front_leg"),
            self._mjcf_root.find("body", "left_center_leg"),
            self._mjcf_root.find("body", "left_back_leg"),
            self._mjcf_root.find("body", "right_front_leg"),
            self._mjcf_root.find("body", "right_center_leg"),
            self._mjcf_root.find("body", "right_back_leg"),
        )

    @composer.cached_property
    def end_effectors(self):
        return self._leg_bodies

    @composer.cached_property
    def observable_joints(self):
        return [
            actuator.joint for actuator in self.actuators
        ]  # pylint: disable=not-an-iterable

    @composer.cached_property
    def egocentric_camera(self):
        return self._mjcf_root.find("camera", "egocentric")

    def aliveness(self, physics):
        return (physics.bind(self.root_body).xmat[-1] - 1.0) / 2.0

    @composer.cached_property
    def ground_contact_geoms(self):
        leg_geoms = []
        for leg in self._leg_bodies:
            leg_geoms.extend(leg.find_all("geom"))
        return tuple(leg_geoms)

    @property
    def prev_action(self):
        return self._prev_action

    @property
    def appendages_sensors(self):
        return self._appendages_sensors

    @property
    def bodies_pos_sensors(self):
        return self._bodies_pos_sensors

    @property
    def bodies_quats_sensors(self):
        return self._bodies_quats_sensors


class KamigamiObservables(base.WalkerObservables):
    """Observables for the Kamigami."""

    @composer.observable
    def appendages_pos(self):
        """Equivalent to `end_effectors_pos` with the head's position appended."""
        appendages = self._entity.end_effectors
        self._entity.appendages_sensors[:] = []
        for body in appendages:
            self._entity.appendages_sensors.append(
                self._entity.mjcf_model.sensor.add(
                    "framepos",
                    name=body.name + "_appendage",
                    objtype="xbody",
                    objname=body,
                    reftype="xbody",
                    refname=self._entity.root_body,
                )
            )

        def appendages_ego_pos(physics):
            return np.reshape(
                physics.bind(self._entity.appendages_sensors).sensordata, -1
            )

        return observable.Generic(appendages_ego_pos)

    @composer.observable
    def bodies_quats(self):
        """Orientations of the bodies as quaternions, in the egocentric frame."""
        bodies = self._entity.bodies
        self._entity.bodies_quats_sensors[:] = []
        for body in bodies:
            self._entity.bodies_quats_sensors.append(
                self._entity.mjcf_model.sensor.add(
                    "framequat",
                    name=body.name + "_ego_body_quat",
                    objtype="xbody",
                    objname=body,
                    reftype="xbody",
                    refname=self._entity.root_body,
                )
            )

        def bodies_ego_orientation(physics):
            return np.reshape(
                physics.bind(self._entity.bodies_quats_sensors).sensordata, -1
            )

        return observable.Generic(bodies_ego_orientation)

    @composer.observable
    def bodies_pos(self):
        """Position of bodies relative to root, in the egocentric frame."""
        bodies = self._entity.bodies
        self._entity.bodies_pos_sensors[:] = []
        for body in bodies:
            self._entity.bodies_pos_sensors.append(
                self._entity.mjcf_model.sensor.add(
                    "framepos",
                    name=body.name + "_ego_body_pos",
                    objtype="xbody",
                    objname=body,
                    reftype="xbody",
                    refname=self._entity.root_body,
                )
            )

        def bodies_ego_pos(physics):
            return np.reshape(
                physics.bind(self._entity.bodies_pos_sensors).sensordata, -1
            )

        return observable.Generic(bodies_ego_pos)

    @property
    def proprioception(self):
        return [
            self.joints_pos,
            self.joints_vel,
            self.body_height,
            self.end_effectors_pos,
            self.appendages_pos,
            self.world_zaxis,
            self.bodies_quats,
            self.bodies_pos,
        ] + self._collect_from_attachments("proprioception")
