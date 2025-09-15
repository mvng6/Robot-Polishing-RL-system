# TCP/IP 통신 프로토콜 명세서 (v1.0)

> **문서 목적**: C++(MFC) 로봇 제어기와 Python 강화학습 에이전트 간의 안정적인 TCP/IP 통신을 위한 규칙을 정의합니다. 이 문서는 두 개발자 간의 공식적인 기술 합의서 역할을 합니다.

## 1. 기본 연결 정보

- **Server**: Python 강화학습 에이전트 (`python_rl_agent`)
- **Client**: C++(MFC) 로봇 제어기 (`cpp_controller`)
- **Server IP Address**: `192.168.0.17`
- **Port**: `8888`

---

## 2. 통신 프로토콜 핵심 규칙

- **데이터 형식**: 고정 길이의 바이너리(Binary) 패킷
- **데이터 패딩 (Padding)**: C++ 측에서 `#pragma pack(push, 1)`을 사용하여 메모리 패딩을 비활성화합니다.
- **바이트 순서 (Endianness)**:
  - **네트워크 전송 시에는 빅 엔디안(Big-Endian, Network Byte Order)을 사용합니다.**
  - C++ 클라이언트가 `htons()`, `htonl()` 함수를 통해 리틀 엔디안 데이터를 **빅 엔디안으로 변환하여 전송**합니다.
  - 따라서 **Python 서버는 수신된 데이터를 빅 엔디안으로 해석**해야 합니다.
- **체크섬 (Checksum) 알고리즘**:
  - **CRC-16/MODBUS** 알고리즘을 사용합니다.
  - 체크섬 계산 범위는 `sof` 필드를 제외한 나머지 모든 데이터 필드입니다.

---

## 3. 데이터 패킷 구조

### 3.1. C++ Client → Python Server (로봇 상태 보고)

- **구조체명**: `PythonCommPacket`
- **총 크기**: **29 bytes**
- **SOF (Start of Frame)**: `0xAAAA`

| 필드명               | 데이터 타입        | 크기 (bytes) | 설명                                           |
| :------------------- | :----------------- | :----------- | :------------------------------------------- |
| `sof`                | `unsigned short`   | 2            | 패킷 시작 플래그, 항상 `0xAAAA`                |
| `Current Force`      | `float`            | 4            | 현재의 Z축 방향 접촉력 (단위: N)               |
| `Target Force`       | `float`            | 4            | 목표 Z축 방향 접촉력 (단위: N)                 |
| `Force Error`        | `float`            | 4            | 접촉력 오차 (단위: N)                          |
| `Force Error dot`    | `float`            | 4            | 접촉력 오차의 미분 값 (단위: N)                 |
| `Force Error int`    | `float`            | 4            | 접촉력 오차의 적분 값 (단위: N)                |
| `pidoutput`          | `float`            | 4            | PID 제어 값                                            |
| `Sander Active Flag` | `unsigned char`    | 1            | 샌더 활성화 상태 (0: 비활성, 1: 활성)                    |
| `checksum`           | `unsigned short`   | 2            | `Current Forc`부터 `Sander Active Flag`까지의 CRC-16 값 |

### 3.2. Python Server → C++ Client (RL 에이전트 명령)

- **구조체명**: `RLAgentPacket`
- **총 크기**: **10 bytes**
- **SOF (Start of Frame)**: `0xBBBB`

| 필드명           | 데이터 타입      | 크기 (bytes) | 설명                                      |
| :------------------ | :--------------- | :----------- | :---------------------------------------- |
| `sof`               | `unsigned short` | 2            | 패킷 시작 플래그, 항상 `0xBBBB`             |
| `Residual Pressure` | `float`          | 4            | 강화학습 에이전트로부터 계산된 잔차 공압 값 (단위: MPa) |
| `Message Send Flag` | `unsigned char`  | 1            | 서버(Python)로부터 메시지 수신 확인용 (0: 미수신, 1: 수신) |
| `Episode On Flag`   | `unsigned char`  | 1            | 서버의 강화학습 에피소드 종료 여부 (0: 에피소드 종료 X, 1: 에피소드 종료 O) |
| `checksum`          | `unsigned short` | 2            | `rlVoltageValue`부터 `confirmFlag`까지의 CRC-16 값 |

---

## 4. Python 구현 가이드

Python에서는 `struct` 모듈을 사용하여 C++ 구조체와 동일한 바이너리 데이터를 생성(Packing)하고 해석(Unpacking)할 수 있습니다.

### 4.1. CRC-16/MODBUS 구현

먼저, C++에서 사용하는 것과 동일한 CRC-16/MODBUS 알고리즘을 Python으로 구현합니다. 아래 함수를 사용하세요.

```python
def crc16_modbus(data: bytes) -> int:
    """CRC-16/MODBUS를 계산합니다."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc
```

### 4.2. 패킷 Packing (Python → C++)

`RLAgentPacket`을 생성하여 C++로 전송하는 예제입니다.

```python
import struct

def create_rl_agent_packet(voltage: float, confirm: int) -> bytes:
    """RLAgentPacket을 생성하고 바이너리로 변환합니다."""
    sof = 0xBBBB

    # 1. SOF를 제외한 데이터 부분을 먼저 Packing
    # 포맷 문자열: > (빅 엔디안), f (float), B (unsigned char)
    payload = struct.pack('>fB', voltage, confirm)

    # 2. Payload로 체크섬 계산
    checksum = crc16_modbus(payload)

    # 3. 전체 패킷을 최종적으로 Packing
    # 포맷 문자열: > (빅 엔디안), H (unsigned short), f, B, H
    full_packet = struct.pack('>HfBH', sof, voltage, confirm, checksum)

    return full_packet

# 사용 예시
voltage_to_send = 5.5
confirm_flag = 1
packet_to_send = create_rl_agent_packet(voltage_to_send, confirm_flag)
# client_socket.sendall(packet_to_send)
print(f"전송할 패킷 (9 bytes): {packet_to_send.hex()}")
```

### 4.3. 패킷 Unpacking (C++ → Python)

C++로부터 `PythonCommPacket`을 수신하여 해석하는 예제입니다.

```python
import struct

def unpack_robot_status_packet(data: bytes):
    """수신된 바이너리 데이터를 PythonCommPacket으로 해석합니다."""
    # C++ 구조체의 총 크기는 21 바이트 입니다.
    if len(data) < 21:
        print("에러: 데이터 길이가 너무 짧습니다.")
        return None

    # 1. 전체 패킷을 먼저 Unpack
    # 포맷 문자열: > (빅 엔디안), H(sof), f,f,f,f, B(pidFlag), H(checksum)
    try:
        sof, fz, pressure, voltage, pid_val, pid_flag, received_checksum = \
            struct.unpack('>HffffBH', data)
    except struct.error as e:
        print(f"Unpack 에러: {e}")
        return None

    # 2. 체크섬 검증
    # 체크섬 계산 범위의 데이터만 다시 Packing하여 crc 계산
    payload_to_check = struct.pack('>ffffB', fz, pressure, voltage, pid_val, pid_flag)
    calculated_checksum = crc16_modbus(payload_to_check)

    if received_checksum != calculated_checksum:
        print(f"체크섬 오류! 수신: {received_checksum}, 계산: {calculated_checksum}")
        return None

    if sof != 0xAAAA:
        print(f"SOF 오류! 수신: {hex(sof)}")
        return None

    # 3. 검증 완료된 데이터를 딕셔너리로 반환
    return {
        "contactForceZ": fz,
        "chamberPressure": pressure,
        "chamberVoltage": voltage,
        "pidControlValue": pid_val,
        "pidFlag": pid_flag
    }

# 사용 예시 (C++에서 보낸 21바이트 데이터라고 가정)
# received_data = b'\xaa\xaa\xc1\x45\x70\xa3\x42\x61\x85\x1e\x42\xb4\x49\xba\xc1\x01\xXX\xXX' # 체크섬은 임의값
# status = unpack_robot_status_packet(received_data)
# if status:
#     print(f"수신된 로봇 상태: {status}")
```

---

## 5. 변경 이력

- **v1.0 (2025-08-14)**: 문서 최초 작성.
