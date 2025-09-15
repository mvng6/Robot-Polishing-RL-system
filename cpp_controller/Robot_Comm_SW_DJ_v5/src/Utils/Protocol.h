#pragma once
#include <vector>

// C++ -> Python 전송용 패킷
#pragma pack(push, 1)
struct PythonCommPacket {
	unsigned short	sof;						// 0xAAAA (2 bytes)
	float			RL_currentForceZ;			// 현재의 z방향 접촉력 (4 bytes)
	float			RL_targetForceZ;			// 목표 z방향 접촉력 (4 bytes)
	float           RL_forceZError;				// z방향 접촉력 오차 (4 bytes)
	float           RL_forceZErrordot;			// z방향 접촉력 오차의 미분값 (4 bytes)
	float           RL_forceZErrorintegral;		// z방향 접촉력 오차의 적분값 (4 bytes)
	float           RL_currentPID;				// 현재 공압 챔버 제어 압력값 (4 bytes)
	bool			RL_sanderactiveFlag;		// Sander 동작 여부 플래그 (1 byte)
	unsigned short  checksum;					// 데이터 무결성 검증 (2 bytes)
};
#pragma pack(pop)

// Python -> C++ 수신용 패킷
#pragma pack(push, 1)
struct RLAgentPacket {
	unsigned short	sof;						// 0xBBBB(2 bytes)
	float			RL_ResidualP;				// 서버(RL 에이전트)로부터 계산된 잔차 공압 (4 bytes)
	bool			RL_MessagerecvFlag;			// 서버(RL 에이전트)로부터 메세지 수신 확인용 플래그 (1 byte) - 0: 메세지 수신 X / 1: 메세지 수신 O
	bool			RL_EpisodeFlag;				// 서버(RL 에이전트)에서의 에피소드 동작 여부 확인용 플래그 (1 byte)   - 0: 에피소드 동작 O / 1: 에피소드 동작 X
	bool			RL_EndFlag;					// 서버(RL 에이전트)에서 강화학습 종료 신호 플래그 (1 byte)   - 0: 강화학습 진행 / 1: 강화학습 종료
	unsigned short	checksum;					// 데이터 무결성 검증 (2 bytes)
};
#pragma pack(pop)

// ==========================================================
// 프로토콜 관련 함수 선언
// ==========================================================

// CRC-16 체크섬 계산 함수
unsigned short calculate_crc16(const unsigned char* data, size_t length);

// PythonCommPacket을 전송용 바이트 벡터로 변환 (Packing)
std::vector<char> PackRobotStatus(float current_forceZ, float target_forceZ,
	float error_forceZ, float error_forceZ_dot, float error_forceZ_int,
	float cur_PID_output, bool Sander_Flag);

// 수신된 바이트 데이터를 RLAgentPacket 구조체로 변환 (Unpacking)
// 성공 시 true, 체크섬 오류 등 실패 시 false 반환
bool UnpackRLAgentCommand(const char* buffer, int length, RLAgentPacket& outPacket);