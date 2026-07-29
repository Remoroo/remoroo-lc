"""Reader for the xArm real-time report on TCP port 30000.

The Python SDK only knows ports 30001/30002/30003, and on this firmware those
push at 5.3, 5.3 and 99.7 packets per second.  Port 30000 pushes at 250 Hz -- the
same rate the controller runs at, and the same rate remoroo-lc commands at -- so
it is the only source that can record what the robot actually did rather than a
sample of it.  Measured on the rig: a hand-guided path recorded through the SDK
arrives in 0.09 rad steps every 172 ms.

Layout is UFACTORY's, from "Data description of TCP port":
784-byte frame, big-endian header, little-endian float32 payload.  Offsets below
are 0-indexed into the whole frame INCLUDING its 4-byte length prefix, which is
how the vendor's table is numbered (their byte 1 is our index 0).

Two things this buys beyond rate:

* **The controller's own microsecond timestamp.**  Sample timing then comes from
  the robot rather than from whenever our loop got scheduled, so a late read
  shows up as a late read and not as a robot that moved strangely.
* **Actual joint velocities, measured.**  Differencing positions to get velocity
  amplifies exactly the quantisation the recording is trying to capture.
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

import numpy as np

FRAME_BYTES = 784
_N_JOINT_SLOTS = 7  # the frame always carries 7, whatever the arm has

# 0-indexed offsets into the full frame.
OFF_LENGTH = 0  # U32 big
OFF_TIMESTAMP = 4  # U64 big, microseconds
OFF_STATE_MODE = 12  # U8: bits 0-3 state, 4-7 mode
OFF_TARGET_Q = 32  # 7 x FP32 little, rad
OFF_TARGET_QD = 60
OFF_ACTUAL_Q = 116  # 7 x FP32 little, rad
OFF_ACTUAL_QD = 144  # 7 x FP32 little, rad/s
OFF_ACTUAL_TAU = 228  # 7 x FP32 little, N.m
OFF_TARGET_TCP = 424  # 6 x FP32 little, mm + rad
OFF_ACTUAL_TCP = 472  # 6 x FP32 little, mm + rad
OFF_ACTUAL_TCP_SPEED = 496
OFF_GRIPPER_TYPE = 736  # U8
OFF_GRIPPER_STATE = 737  # U8
OFF_GRIPPER_POS = 738  # INT16 big, mm


@dataclass
class RtSample:
    """One 250 Hz frame, in this package's units (radians, metres, seconds)."""

    timestamp_s: float  # the CONTROLLER's clock, not ours
    state: int
    mode: int
    q: np.ndarray  # (n_joints,) rad
    qd: np.ndarray  # (n_joints,) rad/s
    tau: np.ndarray  # (n_joints,) N.m
    tcp_xyz: np.ndarray  # (3,) metres
    #: Orientation as a ROTATION VECTOR (axis * angle), NOT roll-pitch-yaw.  The
    #: vendor table says only "rad"; read as RPY it is 40 deg wrong, read as a
    #: rotation vector it agrees with the SDK's own pose to 0.19 deg.  Measured on
    #: the rig, both ways, because the difference is invisible until it is not.
    tcp_rotvec: np.ndarray  # (3,) rad
    gripper_pos_mm: float
    gripper_state: int


class XArmRealTime:
    """Streaming reader for one arm's port-30000 report."""

    def __init__(self, host: str, n_joints: int = 6, port: int = 30000,
                 timeout: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.n_joints = int(n_joints)
        self._sock: socket.socket | None = None
        self._timeout = timeout
        self._buf = b""

    def __enter__(self) -> XArmRealTime:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def connect(self) -> None:
        self._sock = socket.create_connection((self.host, self.port), timeout=self._timeout)
        self._sock.settimeout(self._timeout)
        # TCP_NODELAY: these are small frames at 250 Hz and Nagle would batch them
        # into bursts, which is exactly the jitter this reader exists to avoid.
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._buf = b""

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    # ------------------------------------------------------------------ #
    def _frame(self) -> bytes:
        """One complete frame, resynchronising on the length field if needed."""
        if self._sock is None:
            raise RuntimeError(f"{self.host}: not connected")
        while True:
            while len(self._buf) < 4:
                self._buf += self._recv()
            declared = struct.unpack_from(">I", self._buf, 0)[0]
            if declared != FRAME_BYTES:
                # Lost alignment.  Drop a byte and try again rather than
                # returning a plausible-looking frame decoded from garbage.
                self._buf = self._buf[1:]
                continue
            while len(self._buf) < declared:
                self._buf += self._recv()
            frame, self._buf = self._buf[:declared], self._buf[declared:]
            return frame

    def _recv(self) -> bytes:
        chunk = self._sock.recv(65536)
        if not chunk:
            raise ConnectionError(f"{self.host}: report stream closed")
        return chunk

    def read(self) -> RtSample:
        f = self._frame()
        n = self.n_joints
        q = np.asarray(struct.unpack_from(f"<{_N_JOINT_SLOTS}f", f, OFF_ACTUAL_Q)[:n])
        qd = np.asarray(struct.unpack_from(f"<{_N_JOINT_SLOTS}f", f, OFF_ACTUAL_QD)[:n])
        tau = np.asarray(struct.unpack_from(f"<{_N_JOINT_SLOTS}f", f, OFF_ACTUAL_TAU)[:n])
        tcp = np.asarray(struct.unpack_from("<6f", f, OFF_ACTUAL_TCP))
        sm = f[OFF_STATE_MODE]
        return RtSample(
            timestamp_s=struct.unpack_from(">Q", f, OFF_TIMESTAMP)[0] * 1e-6,
            state=sm & 0x0F,
            mode=(sm >> 4) & 0x0F,
            q=q,
            qd=qd,
            tau=tau,
            tcp_xyz=tcp[:3] / 1000.0,
            tcp_rotvec=tcp[3:],
            gripper_pos_mm=float(struct.unpack_from(">h", f, OFF_GRIPPER_POS)[0]),
            gripper_state=f[OFF_GRIPPER_STATE],
        )

    def drain(self) -> RtSample:
        """The most recent frame, discarding any backlog.

        A control loop wants the newest state, not the oldest queued one; without
        this the reader hands back progressively staler samples whenever the loop
        falls behind, and the staleness never recovers.
        """
        latest = self.read()
        self._sock.setblocking(False)
        try:
            while True:
                try:
                    chunk = self._sock.recv(65536)
                except (BlockingIOError, InterruptedError):
                    break
                if not chunk:
                    break
                self._buf += chunk
            while len(self._buf) >= FRAME_BYTES:
                latest = self.read()
        finally:
            self._sock.setblocking(True)
            self._sock.settimeout(self._timeout)
        return latest
