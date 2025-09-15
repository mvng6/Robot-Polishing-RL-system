#include "pch.h" // MFC 프로젝트의 프리컴파일 헤더
#include "Protocol.h"
#include <winsock2.h> // htons, ntohs, htonl, ntohl 함수 사용을 위해 추가

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
	float cur_PID_output, bool Sander_Flag)
{
	PythonCommPacket packet;
	packet.sof = 0xAAAA;									// Start of Frame (2바이트)
	packet.RL_currentForceZ = current_forceZ;				// 현재 z방향 접촉력 (4바이트)
	packet.RL_targetForceZ = target_forceZ;					// 목표 z방향 접촉력 (4바이트)
	packet.RL_forceZError = error_forceZ;					// z방향 접촉력 오차 (4바이트)
	packet.RL_forceZErrordot = error_forceZ_dot;			// z방향 접촉력 오차 미분값 (4바이트)
	packet.RL_forceZErrorintegral = error_forceZ_int;		// z방향 접촉력 오차 적분값 (4바이트)
	packet.RL_currentPID = cur_PID_output;					// 현재 PID 출력값 (4바이트)
	packet.RL_sanderactiveFlag = Sander_Flag;				// Sander 활성화 플래그 (1바이트)

	// 바이트 순서 변환
	packet.sof = htons(packet.sof);
	*(unsigned long*)&packet.RL_currentForceZ = htonl(*(unsigned long*)&packet.RL_currentForceZ);
	*(unsigned long*)&packet.RL_targetForceZ = htonl(*(unsigned long*)&packet.RL_targetForceZ);
	*(unsigned long*)&packet.RL_forceZError = htonl(*(unsigned long*)&packet.RL_forceZError);
	*(unsigned long*)&packet.RL_forceZErrordot = htonl(*(unsigned long*)&packet.RL_forceZErrordot);
	*(unsigned long*)&packet.RL_forceZErrorintegral = htonl(*(unsigned long*)&packet.RL_forceZErrorintegral);
	*(unsigned long*)&packet.RL_currentPID = htonl(*(unsigned long*)&packet.RL_currentPID);

	// 체크섬 계산 및 설정
	packet.checksum = calculate_crc16((const unsigned char*)&packet, sizeof(packet) - sizeof(unsigned short));
	packet.checksum = htons(packet.checksum);				// 체크섬 (2바이트)

	// 실제 전송할 데이터 크기 출력 (디버깅용)
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

	// 원본 데이터를 복사하여 사용 (원본 버퍼를 직접 수정하지 않기 위해)
	RLAgentPacket received_packet = *reinterpret_cast<const RLAgentPacket*>(buffer);

	// 1. 체크섬 검증
	unsigned short received_checksum = ntohs(received_packet.checksum);
	unsigned short calculated_checksum = calculate_crc16(
		(const unsigned char*)&received_packet,
		sizeof(RLAgentPacket) - sizeof(unsigned short)
	);

	if (received_checksum != calculated_checksum) {
		TRACE("Checksum error in Protocol Unpacking!\n");
		return false; // 체크섬 불일치 시 실패
	}

	// 2. 바이트 순서 변환 (Network to Host)
	outPacket.sof = ntohs(received_packet.sof);

	float rl_Pressure = received_packet.RL_ResidualP;
	*(unsigned long*)&rl_Pressure = ntohl(*(unsigned long*)&rl_Pressure);
	outPacket.RL_ResidualP = rl_Pressure;

	outPacket.RL_MessagerecvFlag = received_packet.RL_MessagerecvFlag;
	outPacket.RL_EpisodeFlag = received_packet.RL_EpisodeFlag;
	outPacket.RL_EndFlag = received_packet.RL_EndFlag;
	outPacket.checksum = received_checksum;

	return true;
}