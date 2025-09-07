#pragma once
#define NOMINMAX

// 1) 표준 라이브러리 헤더
#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <algorithm>
#include <cstddef>
#include <queue>
#include <fstream>
#include <vector>
#include <afxsock.h>

// 2) 외부 SDK 헤더
#include <vcisdk.h>
#include <NIDAQmx.h>

// 3) Windows 헤더
#include <windows.h>
#include <atlconv.h>

// 4) 프로젝트-로컬 헤더
#include "src/Math/AllMath.h"
#include "src/Robot/AllRobot.h"
#include "src/Utils/AllUtils.h"
#include "src/Pneumatics/AirControl.h"
#include "src/FTSensor/AllFTSensor.h"

// 5) 사용자 정의 메세지 ID
#define WM_USER_TCP_DISCONNECT (WM_USER + 1)

// Flags 구조체 정의
struct Flags {
	std::atomic<bool> realtimeConnected{ false };			// 로봇 연결 여부
	std::atomic<bool> servoRunning{ false };				// 서보 제어 활성화 여부
	std::atomic<bool> ftRunning{ false };					// FT 센서 활성화 여부
	std::atomic<bool> robotRunning{ false };				// 로봇 구동 쓰레드 활성화 여부
	std::atomic<bool> airTask{ false };						// 공압 제어 Task 활성화 여부
	std::atomic<bool> airRunning{ false };					// 공압 제어 쓰레드 내부 루프실행 여부	
	std::atomic<bool> logThreadRunning{ false };			// 로그 쓰레드 활성화 플래그
	std::atomic<bool> loadTraj{ false };					// false: 평면, true = 곡면
	std::atomic<bool> flat_stop{ false };					// 평면 구동 정지 플래그
	std::atomic<bool> curve_stop{ false };					// 곡면 구동 정지 플래그

	std::atomic<bool> tcpip_flag{ false };					// TCP/IP 통신 연결 여부
	std::atomic<bool> RL_pid_flag{ false };					// RL PID 제어 활성화 여부
	std::atomic<bool> RL_sanderactive_flag{ false };		// RL Sander 활성화 여부
};

// 설정 구조체 정의
struct Setting {
	std::atomic<float> Force_limit_N{};						// 힘 한계 [N]
	std::atomic<float> Target_Force_N{};					// 목표 힘 [N]
	std::atomic<float> Target_vz{};							// 목표 속도 [mm/s]
	std::atomic<float> vx_mms{};							// x방향 로봇 명령 속도 [mm/s]
	std::atomic<float> vy_mms{};							// y방향 로봇 명령 속도 [mm/s]
	std::atomic<float> vz_mms{};							// z방향 로봇 명령 속도 [mm/s]
	
	std::array<float, 2> x_pos_bound{ -255.0f, -100.0f };	// 평면 구동 시 x축 구동 위치(시작, 종료) [mm] => 사용자 환경에 맞춰 위치 수정 적용 필요 <=
	
	std::atomic<float> Contact_time{};						// 접촉 시간 [s]

	std::atomic<float> set_Hz = 1000.0f;					// while 루프 제어 주기 [Hz]
	std::atomic<float> set_Hz_servo = 1000.0f;				// 서보 제어 주기 [Hz] (곡면 구동의 경우 500Hz로 동작하기 때문에 개별적으로 설정 필요)
	float get_set_ns() const {								// 나노초 단위 [ns]
		return 1e9f / set_Hz.load();
	}

	int Control_Step = 0;									// 평면 구동 시 제어 스텝 카운터
	std::atomic<bool> First_Contact{ false };				// 평면 구동 시 제어 스텝(Control_Step) 첫 실행 여부 확인용
};

// 로봇 제어 관련 구조체 정의
struct ServoControlData {
	// 평면 경로 구동용
	std::array<float, 6> servo_flange_pos_des_key{};		// 평면 경로 구동용 Flange 위치 목표값 [mm, rad]
	std::array<float, 6> servo_flange_vel_des_key{};		// 평면 경로 구동용 Flange 속도 목표값 [mm/s, rad/s]
	std::array<float, 6> servo_flange_acc_des_key{};		// 평면 경로 구동용 Flange 가속도 목표값 [mm/s^2, rad/s^2]

	// 곡면 경로 구동용
	std::array<float, 6> servo_tcp_pos_des_key{};			// 곡면 경로 구동용 TCP 위치 목표값 [mm, rad]
	std::array<float, 6> servo_tcp_vel_des_key{};			// 곡면 경로 구동용 TCP 속도 목표값 [mm/s, rad/s]
	std::array<float, 6> servo_tcp_acc_des_key{};			// 곡면 경로 구동용 TCP 가속도 목표값 [mm/s^2, rad/s^2]

	std::atomic<float> vx_cmd{};							// 로봇의 x방향 속도 제어를 위한 공유 변수
	std::atomic<float> vy_cmd{};							// 로봇의 y방향 속도 제어를 위한 공유 변수
	std::atomic<float> vz_cmd{};							// 로봇의 z방향 속도 제어를 위한 공유 변수	
	std::atomic<float> roll_cmd{};							// 로봇의 Roll 속도 제어를 위한 공유 변수 [rad/s]
	std::atomic<float> pitch_cmd{};							// 로봇의 Pitch 속도 제어를 위한 공유 변수 [rad/s]
	std::atomic<float> yaw_cmd{};							// 로봇의 Yaw 속도 제어를 위한 공유 변수 [rad/s]

	std::array<float, 2> jacc{};							// 평면 경로 구동 시 가속도 공유 변수
	std::array<float, 6> home_servoj_vel{};					// 로봇의 Home 위치 구동 시 관절 속도 제어를 위한 공유 변수 [rad/s]	
	std::array<float, 6> home_servoj_acc{};					// 로봇의 Home 위치 구동 시 관절 가속도 제어를 위한 공유 변수 [rad/s^2]
};

// 데이터 저장 구조체 정의
struct LogData {
	std::vector<float> data;								// 로봇 상태 및 센서 데이터 저장용 배열
};

struct TCPIP {
	std::atomic<float> rl_pressure_from_server{ 0.0f };		// RL 에이전트로부터 수신된 공압 데이터
	std::atomic<bool> is_new_message_received{ false };		// RL 에이전트로부터 새로운 메세지 수신 여부
	std::atomic<bool> episode_state_flag{ false };		// RL 에이전트로부터 에피소드 상태 수신 여부
};


// CRobotCommSWDJv5Dlg 대화 상자
class CRobotCommSWDJv5Dlg : public CDialogEx
{
public:
	CRobotCommSWDJv5Dlg(CWnd* pParent = nullptr);			// 표준 생성자

	RobotState::Snapshot m_robotDataSnapshot;				// 로봇 상태 스냅샷

	std::queue<LogData> m_logQueue;							// 로그 데이터 저장용 큐
	std::condition_variable m_logCv;						// 로그 쓰레드가 대기할 때 사용하는 조건 변수
	std::ofstream m_dataFile;								// 로그 데이터 파일 스트림

	std::vector<RT_Trajectory> m_traj_data;					// 경로 데이터 저장용 멤버 변수

	// ========================================
	// 쓰레드 함수 선언
	static UINT Thread_Servo(LPVOID pParam);				// 로봇 서보제어 쓰레드
	static UINT Thread_FT_Receive(LPVOID pParam);			// FT 센서를 통한 센서값 획득 쓰레드
	static UINT Thread_AirControl(LPVOID pParam);			// 공압 제어 쓰레드
	static UINT Thread_Contact_Flat(LPVOID pParam);			// 평면 구동 쓰레드
	static UINT Thread_Contact_Curve(LPVOID pParam);		// 곡면 구동 쓰레드
	static UINT Thread_Logger(LPVOID pParam);				// 로그 데이터 저장 쓰레드

	// ================== 테스트 =================
	static UINT Thread_Contact_Flat_RL(LPVOID pParam);		// RL 기반 평면 구동 쓰레드
	// ===========================================
	// 
	// 
	// =========================================
	// 버튼 핸들러
	afx_msg void OnBnClickedButRobotConnect();				// 로봇 연결 버튼 핸들러
	afx_msg void OnBnClickedButRobotDisconnect();			// 로봇 연결 해제 버튼 핸들러
	afx_msg void OnBnClickedButHomeInit();					// 로봇 위치 초기화 버튼 핸들러
	afx_msg void OnBnClickedButHomeMove();					// 로봇 홈 위치 이동 버튼 핸들러	
	afx_msg void OnBnClickedButServo();						// 서보 제어 시작 버튼 핸들러	
	afx_msg void OnBnClickedButFtSensorOn();				// FT 센서 활성화 버튼 핸들러
	afx_msg void OnBnClickedButAirOn();						// 공압 제어 활성화 버튼 핸들러
	afx_msg void OnBnClickedButLoadTrajFlat();				// 평면 경로 불러오기 버튼 핸들러
	afx_msg void OnBnClickedButLoadTrajCurve();				// 곡면 경로 불러오기 버튼 핸들러
	afx_msg void OnBnClickedButForceControl();				// 로봇 구동 & 힘 제어 시작 버튼 핸들러
	afx_msg void OnBnClickedButRobotStop();					// 로봇 정지 버튼 핸들러
	afx_msg void OnTimer(UINT_PTR nIDEvent);				// 타이머 이벤트 핸들러
	afx_msg void OnClose();									// 대화 상자 닫기 핸들러
	afx_msg void OnDestroy();								// 대화 상자 파괴 핸들러
	afx_msg void OnBnClickedButTcpip();						// TCP/IP 연결 버튼 핸들러
	afx_msg void OnBnClickedButTcpsend();					// TCP/IP 데이터 전송 버튼 핸들러

	// =========================================
	// GUI 상태 표시창
	CEdit var_status_gui;	

	// ========================================
	// 로봇 제어 과정에서 발생하는 감지 에러 핸들러
	void HandleServoError(int servo_msg, const CString& errorType);

	// TCP 데이터 수신 시 호출될 콜백 핸들러 함수 선언
	void OnTcpDataReceived(const CString& message);
	void OnRlDataReceived(const RLAgentPacket& packet);

	// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ROBOT_COMM_SW_DJ_V5_DIALOG };
#endif

protected:
	// 생성된 메시지 맵 함수
	virtual BOOL OnInitDialog();
	virtual void DoDataExchange(CDataExchange* pDX);		// DDX/DDV 지원
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();	
	DECLARE_MESSAGE_MAP()
	afx_msg LRESULT OnTcpDisconnect(WPARAM wParam, LPARAM lParam);	// TCP 연결 해제 메시지 핸들러 함수 선언
	
private:
	// ===============================
	// 쓰레드 관리 함수
	bool StartServoThread();								// 서보 제어 쓰레드 시작
	bool StartFTThread();									// FT 센서 쓰레드 시작
	bool StartAirThread();									// 공압 제어 쓰레드 시작
	bool StartContactThread_Flat();							// 평면 구동 쓰레드 시작
	bool StartContactThread_Curve();						// 곡면 구동 쓰레드 시작
	void QuitAllThreads();									// 모든 쓰레드 종료

	// ================== 테스트 =================
	bool StartContactThread_Flat_RL();						// RL 기반 평면 구동 쓰레드 시작 (테스트)
	// ===========================================


	// ===============================
	// 로봇 · 센서 상태 멤버
	RobotState			m_robotState;						// 로봇 상태 관리	
	FTSensorFunction	m_ftSensor;							// FT 센서 기능 관리
	Flags				m_flags;							// 플래그 관리
	Setting				m_setting;							// 각종 설정 관리

	// ===============================
	// 서보 · 공압 제어용 변수
	ServoControlData	m_servoctrl;						// 서보 제어에 사용되는 변수
	AirControl			m_airctrl;							// 공압 제어에 사용되는 변수
	PIDController		m_pidctrl;							// 힘 제어를 위한 PID 컨트롤러
	VelocityProfiler	m_velProfile;						// 로봇 하강 & 상승 속도 프로파일 관리

	// ===============================
	// TCP 통신
	TcpClient m_tcpClient;
	TCPIP m_tcpip;										// TCP/IP 통신 관련 변수
	CString m_tcpReceivedData;
	std::mutex m_tcpMutex;

	// Python으로부터 받은 값을 저장할 멤버 변수
	std::atomic<float> m_received_RL_Pressure{ 0.0f };		// RL 잔차 압력 값 [MPa]
	std::atomic<bool> m_received_RL_Confirm_Flag{ false };	// RL 메시지 수신 플래그
	std::atomic<bool> m_received_RL_Episode_Flag{ false };	// RL Episode Flag

	// ===============================
	//  쓰레드 핸들
	// ===============================
	CWinThread* m_pThread_Servo = nullptr;					// 서보 제어 쓰레드
	CWinThread* m_pThread_FT = nullptr;						// FT 센서 쓰레드
	CWinThread* m_pThread_FC = nullptr;						// 접촉 쓰레드
	CWinThread* m_pThread_Air = nullptr;					// 공압 제어 쓰레드
	CWinThread* m_pThread_Logger = nullptr;					// 로그 쓰레드
	HANDLE      hThread_Servo = nullptr;					// 서보 제어 쓰레드 핸들
	HANDLE      hThread_FT = nullptr;						// FT 센서 쓰레드 핸들
	HANDLE      hThread_FC = nullptr;						// 접촉 쓰레드 핸들
	HANDLE      hThread_Air = nullptr;						// 공압 제어 쓰레드 핸들
	HANDLE      hThread_Logger = nullptr;					// 데이터 로그 쓰레드 핸들
	
	std::mutex m_robotMutex;								// 로봇 상태 보호용 뮤텍스
	std::mutex m_logMutex;									// 로그 데이터 보호용 뮤텍스	
	std::mutex m_trajMutex;									// 경로 데이터 보호용 뮤텍스

	HICON m_hIcon;											// 대화 상자 아이콘 핸들

	void UpdateGUI();										// GUI 업데이트 함수

	COScopeCtrl* _rtGraphforce = nullptr;					// 힘 그래프
	COScopeCtrl* _rtGraphpos = nullptr;						// 위치 그래프

	double m_pidInitialOutput_N = 0.0;						// PID 초기 출력값 [N]

	// GUI용 변수
	double var_freq_gui;									// GUI 업데이트 주기 [Hz]
	double var_freq_servo;									// 서보 제어 주기 [Hz]
	double var_freq_ft_sensor;								// FT 센서 주기 [Hz]
	double var_freq_main;									// 메인 루프 주기 [Hz]
	
	double var_pos_gui[6] = {};								// 현재 로봇 TCP 위치 [mm, rad]
	double var_ang_gui[6] = {};								// 현재 로봇 각도 [rad]
	double var_ft_gui[6] = {};								// 현재 FT 센서 값 [N, Nm]

	float val_x_vel_current_gui;							// 현재 TCP x방향 속도 [mm/s]
	float val_y_vel_current_gui;							// 현재 TCP y방향 속도 [mm/s]	
	float val_z_vel_current_gui;							// 현재 TCP z방향 속도 [mm/s]

	double var_target_force_gui;							// 목표 힘 [N]
	double var_contact_timer_gui;							// 접촉 시간 [s]

	double var_chamber_air_gui;								// 공압 챔버 압력 [MPa]
	double var_chamber_volt_gui;							// 공압 챔버 전압 [V]
	double var_spindle_air_gui;								// 스핀들 압력 [MPa]
	double var_spindle_volt_gui;							// 스핀들 전압 [V]

	double var_RL_Pressure;									// 서버(RL 에이전트)로부터 받은 잔차 압력 값 [MPa]
};
