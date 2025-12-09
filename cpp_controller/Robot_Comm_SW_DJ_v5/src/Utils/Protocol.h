#pragma once
#include <vector>

// C++ -> Python 통신용 패킷
#pragma pack(push, 1)
struct PythonCommPacket {
	unsigned short	sof;						// 0xAAAA (2 bytes)
	float			RL_currentForceZ;			// 현재 z방향 접촉력 (4 bytes)
	float			RL_targetForceZ;			// 목표 z방향 접촉력 (4 bytes)
	float           RL_forceZError;				// z방향 접촉력 오차 (4 bytes)
	float           RL_forceZErrordot;			// z방향 접촉력 오차의 미분값 (4 bytes)
	float           RL_forceZErrorintegral;		// z방향 접촉력 오차의 적분값 (4 bytes)
	float           RL_currentPID;				// 현재 PID 제어기 출력값 (4 bytes)
	bool			RL_sanderactiveFlag;		// Sander 활성화 상태 플래그 (1 byte)
	unsigned short  checksum;					// 체크섬 검증용 값 (2 bytes)
};
#pragma pack(pop)

// Python -> C++ 수신용 패킷
#pragma pack(push, 1)
struct RLAgentPacket {
	unsigned short	sof;						// 0xBBBB(2 bytes)
	float			RL_gain_P;					// PID P 게인값 (4 bytes)
	float			RL_gain_I;					// PID I 게인값 (4 bytes)
	float			RL_gain_D;					// PID D 게인값 (4 bytes)
	bool			RL_timing_accurate;			// 서버(RL 에이전트)로부터 메시지 수신 확인 플래그 (1 byte) - 0: 메시지 수신 X / 1: 메시지 수신 O
	bool			RL_episode_done;			// 서버(RL 에이전트)에서 에피소드 종료 신호 확인 플래그 (1 byte)   - 0: 에피소드 계속 O / 1: 에피소드 종료 X
	bool			RL_learning_done;			// 서버(RL 에이전트)에서 학습 종료 신호 플래그 (1 byte)   - 0: 학습 종료 / 1: 학습 계속
	unsigned short	checksum;					// 체크섬 검증용 값 (2 bytes)
};
#pragma pack(pop)

// ==========================================================
// 통신 프로토콜 관련 함수 선언
// ==========================================================

// CRC-16 체크섬 계산 함수
unsigned short calculate_crc16(const unsigned char* data, size_t length);

// PythonCommPacket을 바이트 스트림으로 변환 (Packing)
std::vector<char> PackRobotStatus(float current_forceZ, float target_forceZ,
	float error_forceZ, float error_forceZ_dot, float error_forceZ_int,
	float cur_PID_output, bool Sander_Flag);

// 수신된 바이트 데이터를 RLAgentPacket 구조체로 변환 (Unpacking)
// 성공 시 true, 체크섬 오류 시 false 반환
bool UnpackRLAgentCommand(const char* buffer, int length, RLAgentPacket& outPacket);