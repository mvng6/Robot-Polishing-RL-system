#include "pch.h" // MFC 프로젝트에서 사용하는 프리컴파일 헤더
#include "Protocol.h"
#include <winsock2.h> // htons, ntohs, htonl, ntohl 함수 사용을 위해 추가
#include <cstring>
#include <vector>
#include <cstdint>    // uint32_t 타입 사용을 위해 추가

#pragma comment(lib, "ws2_32.lib") // winsock2 라이브러리 링크

// CRC-16 체크섬 계산 함수
unsigned short calculate_crc16(const unsigned char* data, size_t length)
{
	unsigned short crc = 0xFFFF;
	for (size_t i = 0; i < length; i++) {
		crc ^= (unsigned short)data[i];
		for (int j = 0; j < 8; j++) {
			if (crc & 0x0001) {
				crc = (crc >> 1) ^ 0xA001; // CRC-16/MODBUS
			}
			else {
				crc = crc >> 1;
			}
		}
	}
	return crc;
}

// Packing 함수
std::vector<char> PackRobotStatus(float current_forceZ, float target_forceZ, float error_forceZ, float error_forceZ_dot, float error_forceZ_int,
	float cur_PID_output, bool Sander_Flag, float precharge_applied, float j3, bool prep_flag)
{
	PythonCommPacket packet;
	packet.sof = 0xAAAA;									// Start of Frame (2바이트)
	packet.RL_currentForceZ = current_forceZ;				// 현재 z방향 접촉력 (4바이트)
	packet.RL_targetForceZ = target_forceZ;					// 목표 z방향 접촉력 (4바이트)
	packet.RL_forceZError = error_forceZ;					// z방향 접촉력 오차 (4바이트)
	packet.RL_forceZErrordot = error_forceZ_dot;			// z방향 접촉력 오차의 미분값 (4바이트)
	packet.RL_forceZErrorintegral = error_forceZ_int;		// z방향 접촉력 오차의 적분값 (4바이트)
	packet.RL_currentPID = cur_PID_output;					// 현재 PID 출력값 (4바이트)
	packet.RL_sanderactiveFlag = Sander_Flag;				// Sander 활성화 플래그 (1바이트)
	packet.RL_preccharge_applied = precharge_applied;		// 초기 공압 (4바이트)
	packet.RL_j3 = j3;										// 로봇 z 위치좌표 (4바이트)
	packet.RL_prep_flag = prep_flag;						// 샌더 동작 3초전 알림 플래그 (1바이트)

	// 네트워크 바이트 순서 변환 (Host to Network)
	packet.sof = htons(packet.sof);
	*(unsigned long*)&packet.RL_currentForceZ = htonl(*(unsigned long*)&packet.RL_currentForceZ);
	*(unsigned long*)&packet.RL_targetForceZ = htonl(*(unsigned long*)&packet.RL_targetForceZ);
	*(unsigned long*)&packet.RL_forceZError = htonl(*(unsigned long*)&packet.RL_forceZError);
	*(unsigned long*)&packet.RL_forceZErrordot = htonl(*(unsigned long*)&packet.RL_forceZErrordot);
	*(unsigned long*)&packet.RL_forceZErrorintegral = htonl(*(unsigned long*)&packet.RL_forceZErrorintegral);
	*(unsigned long*)&packet.RL_currentPID = htonl(*(unsigned long*)&packet.RL_currentPID);
	*(unsigned long*)&packet.RL_preccharge_applied = htonl(*(unsigned long*)&packet.RL_preccharge_applied);
	*(unsigned long*)&packet.RL_j3 = htonl(*(unsigned long*)&packet.RL_j3);

	// 체크섬 계산 및 설정
	packet.checksum = calculate_crc16((const unsigned char*)&packet, sizeof(packet) - sizeof(unsigned short));
	packet.checksum = htons(packet.checksum);				// 체크섬 (2바이트)

	// 바이트 스트림으로 변환하여 반환 (송신용)
	std::vector<char> result(reinterpret_cast<const char*>(&packet),
		reinterpret_cast<const char*>(&packet) + sizeof(packet));

	return result;
}

// Unpacking 함수
bool UnpackRLAgentCommand(const char* buffer, int length, RLAgentPacket& outPacket)
{
	if (length < sizeof(RLAgentPacket)) {
		return false;
	}

	// 수신 데이터를 구조체로 복사 (체크섬 검증을 위해 원본 데이터 보존)
	RLAgentPacket received_packet = *reinterpret_cast<const RLAgentPacket*>(buffer);

	// 1. 체크섬 검증
	unsigned short received_checksum = ntohs(received_packet.checksum);
	unsigned short calculated_checksum = calculate_crc16(
		(const unsigned char*)&received_packet,
		sizeof(RLAgentPacket) - sizeof(unsigned short)
	);

	if (received_checksum != calculated_checksum) {
		TRACE("Checksum error in Protocol Unpacking!\n");
		return false; // 체크섬 불일치로 실패
	}

	// 2. 네트워크 바이트 순서 변환 (Network to Host)
	outPacket.sof = ntohs(received_packet.sof);

	// 초기 공압값의 네트워크 바이트 순서 변환 (Network to Host)
	float rl_precharge = received_packet.RL_precharge;
	*(uint32_t*)&rl_precharge = ntohl(*(uint32_t*)&rl_precharge);

	// PID 게인값들의 네트워크 바이트 순서 변환 (Network to Host)
	float rl_gain_P = received_packet.RL_gain_P;
	*(uint32_t*)&rl_gain_P = ntohl(*(uint32_t*)&rl_gain_P);
	
	float rl_gain_I = received_packet.RL_gain_I;
	*(uint32_t*)&rl_gain_I = ntohl(*(uint32_t*)&rl_gain_I);
	
	float rl_gain_D = received_packet.RL_gain_D;
	*(uint32_t*)&rl_gain_D = ntohl(*(uint32_t*)&rl_gain_D);

	outPacket.RL_precharge = rl_precharge;
	outPacket.RL_gain_P = rl_gain_P;
	outPacket.RL_gain_I = rl_gain_I;
	outPacket.RL_gain_D = rl_gain_D;

	outPacket.RL_timing_accurate = received_packet.RL_timing_accurate;
	outPacket.RL_episode_done = received_packet.RL_episode_done;
	outPacket.RL_learning_done = received_packet.RL_learning_done;
	outPacket.checksum = received_checksum;

	return true;
}