"""
CRC32 computation for Unitree Go2 LowCmd messages.

Ports the C++ implementation from motor_crc.cpp / motor_crc.h.
The CRC must be computed over the raw C struct byte layout.
"""

import struct

# Go2 LowCmd C struct layout (812 bytes total):
#
#   [0..3]     head[2], levelFlag, frameReserve           4 bytes
#   [4..11]    SN[2]                                      8 bytes
#   [12..19]   version[2]                                 8 bytes
#   [20..21]   bandwidth                                  2 bytes
#   [22..23]   padding (align MotorCmd to 4 bytes)        2 bytes
#   [24..743]  motorCmd[20] × 36 bytes                  720 bytes
#   [744..747] BmsCmd {off, reserve[3]}                   4 bytes
#   [748..787] wirelessRemote[40]                        40 bytes
#   [788..799] led[12]                                   12 bytes
#   [800..801] fan[2]                                     2 bytes
#   [802]      gpio                                       1 byte
#   [803]      padding                                    1 byte
#   [804..807] reserve                                    4 bytes
#   [808..811] crc                                        4 bytes

_HEADER_FMT = '<2sBBIIIIH'     # 22 bytes (head[2], levelFlag, frameReserve, SN[2], version[2], bandwidth)
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 22
_HEADER_PAD = 2                              # padding to align motorCmd

_MOTOR_FMT = '<Bxxxfffff3I'    # 36 bytes (mode, 3pad, q, dq, tau, kp, kd, reserve[3])
_MOTOR_SIZE = struct.calcsize(_MOTOR_FMT)    # 36

_BMS_FMT = '<B3s'              # 4 bytes (off, reserve[3])
_BMS_SIZE = struct.calcsize(_BMS_FMT)        # 4

_TOTAL_SIZE = 812
_CRC_WORDS = (_TOTAL_SIZE - 4) // 4  # 202 uint32 words (exclude crc field)

_POLYNOMIAL = 0x04C11DB7


def _crc32_core(words):
    """Compute CRC32 over a sequence of uint32 words.

    Direct port of crc32_core() from motor_crc.cpp.
    """
    crc = 0xFFFFFFFF
    for word in words:
        xbit = 1 << 31
        data = word
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ _POLYNOMIAL
            else:
                crc = (crc << 1) & 0xFFFFFFFF
            if data & xbit:
                crc ^= _POLYNOMIAL
            xbit >>= 1
    return crc


def compute_crc(msg):
    """Compute and set the CRC field on a unitree_go/msg/LowCmd message.

    Packs the message into the C struct binary layout, computes CRC32,
    and sets msg.crc.
    """
    buf = bytearray(_TOTAL_SIZE)
    offset = 0

    # Header
    struct.pack_into(
        _HEADER_FMT, buf, offset,
        bytes(msg.head), msg.level_flag, msg.frame_reserve,
        msg.sn[0], msg.sn[1],
        msg.version[0], msg.version[1],
        msg.bandwidth,
    )
    offset = _HEADER_SIZE + _HEADER_PAD  # 24

    # 20 motor commands
    for i in range(20):
        mc = msg.motor_cmd[i]
        struct.pack_into(
            _MOTOR_FMT, buf, offset,
            mc.mode, mc.q, mc.dq, mc.tau, mc.kp, mc.kd,
            mc.reserve[0], mc.reserve[1], mc.reserve[2],
        )
        offset += _MOTOR_SIZE

    # BmsCmd
    struct.pack_into(
        _BMS_FMT, buf, offset,
        msg.bms_cmd.off, bytes(msg.bms_cmd.reserve),
    )
    offset += _BMS_SIZE

    # wirelessRemote[40]
    buf[offset:offset + 40] = bytes(msg.wireless_remote)
    offset += 40

    # led[12]
    buf[offset:offset + 12] = bytes(msg.led)
    offset += 12

    # fan[2]
    buf[offset:offset + 2] = bytes(msg.fan)
    offset += 2

    # gpio + 1 byte padding
    buf[offset] = msg.gpio
    offset += 2  # gpio(1) + padding(1)

    # reserve
    struct.pack_into('<I', buf, offset, msg.reserve)
    offset += 4

    # Interpret first 808 bytes as 202 uint32 words (little-endian)
    words = struct.unpack_from(f'<{_CRC_WORDS}I', buf, 0)

    msg.crc = _crc32_core(words)
