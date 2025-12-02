///////////////////////////////////////////////////////////////////////////////
//
// Robot_Comm_SW_DJ_v5Dlg.cpp
// 
///////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////
// 1. 헤더 파일 및 전역 변수 정의
/////////////////////////////////////////////
// 1) 프리컴파일 헤더
#include "pch.h"
#define NOMINMAX
#include <random>

// 2) 외부(SDK) 헤더
#include "DRFLEx.h"

// 3) 프로젝트-로컬 헤더
#include "Robot_Comm_SW_DJ_v5.h"
#include "Robot_Comm_SW_DJ_v5Dlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#pragma comment(linker, "/entry:WinMainCRTStartup /subsystem:console")	// MFT 사용 시 디버그를 위한 콘솔창 띄우는 법
#endif

using namespace DRAFramework;
using std::chrono::system_clock;
using std::chrono::nanoseconds;

CString Status_gui_str = _T("");
CDRFLEx Drfl;

// -----------------------------------------------
// OnTimer 관련
static int i_ont = 0;
clock_t t_check, t_check_old;

// -----------------------------------------------
// 유틸리티 함수: 부동소수점 정밀도 제어
// -----------------------------------------------
/**
 * @brief 부동소수점 값을 지정된 소수점 자리수로 반올림
 * @param value 반올림할 값
 * @param decimals 소수점 자리수 (예: 3이면 소수점 3번째 자리까지)
 * @return 반올림된 값
 */
inline double roundToDecimalPlaces(double value, int decimals) {
	const double multiplier = std::pow(10.0, decimals);
	return std::round(value * multiplier) / multiplier;
}

/////////////////////////////////////////////
// 2. 클래스 생성자 및 초기화 함수
/////////////////////////////////////////////
CRobotCommSWDJv5Dlg::CRobotCommSWDJv5Dlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_ROBOT_COMM_SW_DJ_V5_DIALOG, pParent),
	var_freq_gui(0.0),				// GUI 주기 변수 초기화
	var_freq_servo(0.0),			// 서보 주기 변수 초기화
	var_freq_ft_sensor(0.0),		// FT 센서 주기 변수 초기화
	var_freq_main(0.0),				// 메인 주기 변수 초기화	
	val_x_vel_current_gui(0),		// 현재 x방향 속도 GUI 변수 초기화
	val_y_vel_current_gui(0),		// 현재 y방향 속도 GUI 변수 초기화
	val_z_vel_current_gui(0),		// 현재 z방향 속도 GUI 변수 초기화
	var_target_force_gui(0.0),		// 목표 힘 GUI 변수 초기화
	var_contact_timer_gui(0),		// 접촉 타이머 GUI 변수 초기화
	var_chamber_air_gui(0),			// 공압 챔버 압력 GUI 변수 초기화
	var_chamber_volt_gui(0),		// 공압 챔버 전압 GUI 변수 초기화
	var_spindle_air_gui(0),			// 공압 스핀들 압력 GUI 변수 초기화
	var_spindle_volt_gui(0)			// 공압 스핀들 전압 GUI 변수 초기화	
	, var_gain_p_gui(0)
	, var_gain_i_gui(0)
	, var_gain_d_gui(0)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);  // 기본 아이콘 설정

	for (int i = 0; i < 6; ++i)
	{
		var_pos_gui[i] = 0.0;		// 현재 로봇 위치 초기화
		var_ang_gui[i] = 0.0;		// 현재 로봇 각도 초기화
		var_ft_gui[i] = 0.0;		// 현재 FT 센서 값 초기화
	}
}

void CRobotCommSWDJv5Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_VAR_STATUS, var_status_gui);						// 상태 표시용 컨트롤
}


BEGIN_MESSAGE_MAP(CRobotCommSWDJv5Dlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()	
	ON_WM_TIMER()	
	ON_WM_CLOSE()
	ON_BN_CLICKED(IDC_BUT_ROBOT_CONNECT, &CRobotCommSWDJv5Dlg::OnBnClickedButRobotConnect)			// 로봇 연결 버튼 클릭 이벤트
	ON_BN_CLICKED(IDC_BUT_ROBOT_DISCONNECT, &CRobotCommSWDJv5Dlg::OnBnClickedButRobotDisconnect)	// 로봇 연결 해제 버튼 클릭 이벤트
	ON_BN_CLICKED(IDC_BUT_HOME_INIT, &CRobotCommSWDJv5Dlg::OnBnClickedButHomeInit)					// 홈 초기화 버튼 클릭 이벤트
	ON_BN_CLICKED(IDC_BUT_HOME_MOVE, &CRobotCommSWDJv5Dlg::OnBnClickedButHomeMove)					// 홈 이동 버튼 클릭 이벤트
	ON_BN_CLICKED(IDC_BUT_SERVO, &CRobotCommSWDJv5Dlg::OnBnClickedButServo)							// 서보 제어 버튼 클릭 이벤트
	ON_BN_CLICKED(IDC_BUT_FT_SENSOR_ON, &CRobotCommSWDJv5Dlg::OnBnClickedButFtSensorOn)				// FT 센서 ON 버튼 클릭 이벤트
	ON_BN_CLICKED(IDC_BUT_AIR_ON, &CRobotCommSWDJv5Dlg::OnBnClickedButAirOn)						// 공압 제어 ON 버튼 클릭 이벤트
	ON_BN_CLICKED(IDC_BUT_FORCE_CONTROL, &CRobotCommSWDJv5Dlg::OnBnClickedButForceControl)			// 힘 제어 버튼 클릭 이벤트
	ON_BN_CLICKED(IDC_BUT_LOAD_TRAJ_FLAT, &CRobotCommSWDJv5Dlg::OnBnClickedButLoadTrajFlat)			// 평면 경로 불러오기 버튼 클릭 이벤트
	ON_BN_CLICKED(IDC_BUT_LOAD_TRAJ_CURVE, &CRobotCommSWDJv5Dlg::OnBnClickedButLoadTrajCurve)		// 곡선 경로 불러오기 버튼 클릭 이벤트
	ON_BN_CLICKED(IDC_BUT_ROBOT_STOP, &CRobotCommSWDJv5Dlg::OnBnClickedButRobotStop)				// 로봇 정지 버튼 클릭 이벤트
	ON_BN_CLICKED(IDC_BUT_TCPIP, &CRobotCommSWDJv5Dlg::OnBnClickedButTcpip)							// TCP/IP 연결 버튼 클릭 이벤트
	ON_BN_CLICKED(IDC_BUT_TCPSEND, &CRobotCommSWDJv5Dlg::OnBnClickedButTcpsend)						// TCP/IP 데이터 전송 버튼 클릭 이벤트
	ON_MESSAGE(WM_USER_TCP_DISCONNECT, &CRobotCommSWDJv5Dlg::OnTcpDisconnect)						// TCP 연결 해제 메시지 핸들러
END_MESSAGE_MAP()

// CRobotCommSWDJv5Dlg 메시지 처리기
BOOL CRobotCommSWDJv5Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 이 대화 상자의 아이콘을 설정  
	// 응용 프로그램의 주 창이 대화 상자가 아닐 경우에는 프레임워크가 이 작업을 자동으로 수행
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정

	m_tcpClient.SetReceiveCallback([this](const RLAgentPacket& packet) {
		this->OnRlDataReceived(packet);
		});

	CRect rtGraph_force, rtGraph_pos;														// 그래프 컨트롤의 위치를 저장할 변수

	GetDlgItem(IDC_GRAPH_FORCE)->GetWindowRect(rtGraph_force);								// 그래프 컨트롤의 위치를 가져옴
	GetDlgItem(IDC_GRAPH_POSZ)->GetWindowRect(rtGraph_pos);									// 그래프 컨트롤의 위치를 가져옴
	ScreenToClient(rtGraph_force);															// 화면 좌표를 클라이언트 좌표로 변환
	ScreenToClient(rtGraph_pos);															// 화면 좌표를 클라이언트 좌표로 변환

	// 실시간 접촉력 그래프
	_rtGraphforce = new COScopeCtrl(2);														// 2개의 트렌드를 생성
	_rtGraphforce->Create(WS_VISIBLE | WS_CHILD, rtGraph_force, this, IDC_GRAPH_FORCE);		// 그래프 컨트롤 생성
	_rtGraphforce->SetRanges(0.0, 80.0);													// Y축 범위 조정
	_rtGraphforce->SetYGridSpacing(10.0);													// Y축 그리드 간격을 조정
	_rtGraphforce->autofitYscale = false;													// Y축 자동 스케일링 비활성화
	_rtGraphforce->SetYUnits(CString("Force [N]"));											// Y축 단위 설정
	_rtGraphforce->SetXUnits(CString("Time [s]"));											// X축 단위 설정
	_rtGraphforce->SetLegendLabel(CString("Desired Force"), 0);								// 목표 힘 레이블 설정
	_rtGraphforce->SetLegendLabel(CString("Current Filtered Force"), 1);					// 현재 필터링된 힘 레이블 설정
	_rtGraphforce->SetPlotColor(RGB(255, 0, 0), 0);											// 목표 힘 그래프 색상 설정
	_rtGraphforce->SetPlotColor(RGB(0, 255, 0), 1);											// 현재 필터링된 힘 그래프 색상 설정
	_rtGraphforce->InvalidateCtrl();														// 그래프 컨트롤을 무효화하여 다시 그리도록 함

	// 실시간 위치좌표 그래프
	_rtGraphpos = new COScopeCtrl(2);														// 2개의 트렌드를 생성
	_rtGraphpos->Create(WS_VISIBLE | WS_CHILD, rtGraph_pos, this, IDC_GRAPH_POSZ);			// 그래프 컨트롤 생성
	//_rtGraphpos->SetRanges_pos(0.0, 150.0);													// Y축 범위 조정
	_rtGraphpos->SetRanges_pos(350.0, 450.0);													// Y축 범위 조정
	_rtGraphpos->SetYGridSpacing(10.0);														// Y축 그리드 간격을 조정
	_rtGraphpos->autofitYscale = false;														// Y축 자동 스케일링 비활성화
	_rtGraphpos->SetYUnits(CString("PosZ [mm]"));											// Y축 단위 설정
	_rtGraphpos->SetXUnits(CString("Time [s]"));											// X축 단위 설정
	_rtGraphpos->SetLegendLabel(CString("Desired PosZ"), 0);								// 목표 위치 Z 레이블 설정
	_rtGraphpos->SetLegendLabel(CString("CurrentPosZ"), 1);									// 현재 위치 Z 레이블 설정
	_rtGraphpos->SetPlotColor(RGB(255, 0, 0), 0);											// 목표 위치 Z 그래프 색상 설정
	_rtGraphpos->SetPlotColor(RGB(0, 255, 0), 1);											// 현재 위치 Z 그래프 색상 설정
	_rtGraphpos->InvalidateCtrl();															// 그래프 컨트롤을 무효화하여 다시 그리도록 함

	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환
}

void CRobotCommSWDJv5Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춤
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그림
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서 이 함수를 호출
HCURSOR CRobotCommSWDJv5Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

/////////////////////////////////////////////
// 3. 쓰레드 시작/종료 관리 함수
/////////////////////////////////////////////

// =========================================
// 서보 제어 쓰레드 시작 함수
bool CRobotCommSWDJv5Dlg::StartServoThread()
{
	if (m_pThread_Servo)										// 이미 돌고 있으면 재진입 금지
		return false;

	// 1) 선행 조건: RT-UDP 연결 여부 확인
	if (!m_flags.realtimeConnected.load()) {
		Status_gui_str = _T("Realtime 연결 안되어 있음");
		var_status_gui.SetWindowTextW(Status_gui_str);
		return false;
	}

	// 2) 쓰레드 기동
	m_flags.servoRunning.store(true);							// 서보 제어 루프 ON
	m_pThread_Servo = AfxBeginThread(Thread_Servo, this);		// 서보 제어 쓰레드 시작
	if (!m_pThread_Servo) {										// 쓰레드 생성 실패 처리
		m_flags.servoRunning.store(false);
		Status_gui_str = _T("Servo 쓰레드 생성 실패!");
		var_status_gui.SetWindowTextW(Status_gui_str);
		return false;
	}
	hThread_Servo = m_pThread_Servo->m_hThread;					// 핸들 복사
	
	return true;
}

// =========================================
// FT 센서 쓰레드 시작 함수
bool CRobotCommSWDJv5Dlg::StartFTThread()
{
	if (m_pThread_FT)											// 이미 돌고 있으면 재진입 금지
		return false;

	// 1) FT 센서 연결 여부 확인
	if (!m_ftSensor.isConnected())								// FT 센서가 연결되어 있지 않는 경고 알림 처리
	{
		Status_gui_str = _T("FT 센서가 먼저 연결되어야 합니다!");
		var_status_gui.SetWindowTextW(Status_gui_str);
		return false;
	}

	// 2) FT 센서 쓰레드 시작
	m_flags.ftRunning.store(true);								// FT 센서 루프 ON
	m_pThread_FT = AfxBeginThread(Thread_FT_Receive, this);		// FT 센서 쓰레드 시작
	if (!m_pThread_FT)											// 쓰레드 생성 실패 처리
	{
		m_flags.ftRunning.store(false);
		Status_gui_str = _T("FT 쓰레드 생성 실패!");
		var_status_gui.SetWindowTextW(Status_gui_str);
		return false;
	}
	hThread_FT = m_pThread_FT->m_hThread;						// 핸들 복사

	Status_gui_str = _T("FT 센서 쓰레드 시작 성공");
	var_status_gui.SetWindowTextW(Status_gui_str);
	return true;
}

// =========================================
// 공압 제어 쓰레드 시작 함수
bool CRobotCommSWDJv5Dlg::StartAirThread()
{
	if (m_pThread_Air)											// 이미 돌고 있으면 재진입 금지
		return false;

	// 1) DAQ Task 생성
	if (!m_airctrl.isInitialized())								// 공압 제어가 초기화되지 않은 경우 경고 알림 처리
	{
		if (!m_airctrl.initTasks())
		{
			Status_gui_str = _T("DAQ 초기화 실패. AirControl::Config 설정을 확인하세요.");
			var_status_gui.SetWindowTextW(Status_gui_str);
			return false;
		}
	}

	// 2) 공압 제어 쓰레드 시작
	m_flags.airRunning.store(true);								// 공압 제어 루프 ON
	m_pThread_Air = AfxBeginThread(Thread_AirControl, this);	// 공압 제어 쓰레드 시작
	if (!m_pThread_Air) {										// 쓰레드 생성 실패 처리
		m_flags.airRunning.store(false);
		return false;
	}
	hThread_Air = m_pThread_Air->m_hThread;						// 핸들 복사
	return true;
}

// =========================================
// 로봇 구동 쓰레드 시작 함수 (평면 구동)
bool CRobotCommSWDJv5Dlg::StartContactThread_Flat()
{
	if (m_pThread_FC)											// 이미 돌고 있으면 재진입 금지
		return false;

	// 1) 조건 체크
	if (!m_flags.servoRunning.load())							// 서보 제어가 시작되지 않은 경우 경고 알림 처리
	{
		Status_gui_str = _T("먼저 Servo 제어를 시작하십시오!");
		var_status_gui.SetWindowTextW(Status_gui_str);
		return false;
	}
	if (!m_flags.airRunning.load())								// 공압 제어가 시작되지 않은 경우 경고 알림 처리
	{
		Status_gui_str = _T("먼저 공압 제어를 시작하십시오!");
		var_status_gui.SetWindowTextW(Status_gui_str);
		return false;
	}
	if (!m_flags.ftRunning.load())								// FT 센서가 연결되지 않은 경우 경고 알림 처리
	{
		Status_gui_str = _T("먼저 FT 센서를 연결하십시오!");
		var_status_gui.SetWindowTextW(Status_gui_str);
		return false;
	}

	// 2) 로봇 구동 쓰레드 시작
	m_flags.robotRunning.store(true);							// 로봇 구동 루프 ON
	m_pThread_FC = AfxBeginThread(Thread_Contact_Flat, this);	// 로봇 구동 쓰레드 시작
	if (!m_pThread_FC)											// 쓰레드 생성 실패 처리
	{
		m_flags.robotRunning.store(false);
		Status_gui_str = _T("접촉 쓰레드 시작 실패!");
		var_status_gui.SetWindowTextW(Status_gui_str);
		return false;
	}
	hThread_FC = m_pThread_FC->m_hThread;						// 핸들 복사
	return true;
}

// =========================================
// 로봇 구동 쓰레드 시작 함수 (곡면 구동)
bool CRobotCommSWDJv5Dlg::StartContactThread_Curve()
{
	if (m_pThread_FC)											// 이미 돌고 있으면 재진입 금지
		return false;

	// 1) 선형 조건 체크
	if (!m_flags.servoRunning.load())							// 서보 제어가 시작되지 않은 경우 경고 알림 처리
	{
		Status_gui_str = _T("먼저 Servo 제어를 시작하십시오!");
		var_status_gui.SetWindowTextW(Status_gui_str);
		return false;
	}
	if (!m_flags.ftRunning.load())								// FT 센서가 연결되지 않은 경우 경고 알림 처리
	{
		Status_gui_str = _T("먼저 FT 센서를 연결하십시오!");
		var_status_gui.SetWindowTextW(Status_gui_str);
		return false;
	}
	// 2) 접촉 쓰레드 시작
	m_flags.robotRunning.store(true);							// 로봇 구동 루프 ON
	m_pThread_FC = AfxBeginThread(Thread_Contact_Curve, this);	// 로봇 구동 쓰레드 시작
	if (!m_pThread_FC)											// 쓰레드 생성 실패 처리
	{
		m_flags.robotRunning.store(false);
		Status_gui_str = _T("접촉 쓰레드 시작 실패!");
		var_status_gui.SetWindowTextW(Status_gui_str);
		return false;
	}
	hThread_FC = m_pThread_FC->m_hThread;						// 핸들 복사
	return true;
}

bool CRobotCommSWDJv5Dlg::StartContactThread_Flat_RL()
{
	if (m_pThread_FC)
		return false;

	// 1) 조건 체크
	if (!m_flags.servoRunning.load())							// 서보 제어가 시작되지 않은 경우 경고 알림 처리
	{
		Status_gui_str = _T("먼저 Servo 제어를 시작하십시오!");
		var_status_gui.SetWindowTextW(Status_gui_str);
		return false;
	}
	if (!m_flags.airRunning.load())								// 공압 제어가 시작되지 않은 경우 경고 알림 처리
	{
		Status_gui_str = _T("공압 시스템 연결이 되지 않았으므로 공압 시스템을 연결합니다...");
		var_status_gui.SetWindowTextW(Status_gui_str);
		return false;
	}
	if (!m_flags.ftRunning.load())								// FT 센서가 연결되지 않은 경우 경고 알림 처리
	{
		Status_gui_str = _T("FT 센서 연결이 되지 않았으므로 FT 센서를 연결합니다...");
		var_status_gui.SetWindowTextW(Status_gui_str);
		return false;
	}

	// 2) 로봇 구동 쓰레드 시작
	m_flags.robotRunning.store(true);							// 로봇 구동 루프 ON
	m_pThread_FC = AfxBeginThread(Thread_Contact_Flat_RL, this);	// 로봇 구동 쓰레드 시작

	if (!m_pThread_FC)											// 쓰레드 생성 실패 처리
	{
		m_flags.robotRunning.store(false);
		Status_gui_str = _T("접촉 쓰레드 시작 실패!");
		var_status_gui.SetWindowTextW(Status_gui_str);
		return false;
	}
	hThread_FC = m_pThread_FC->m_hThread;						// 핸들 복사
	return true;
}

// =========================================
// 쓰레드 일괄 종료 함수
void CRobotCommSWDJv5Dlg::QuitAllThreads()
{
	m_flags.realtimeConnected.store(false);										// 서보 쓰레드 구동 시작 조건 종료 플래그 설정
	SafeThreadQuit(m_pThread_Servo, hThread_Servo, m_flags.servoRunning);		// 서보 쓰레드 종료
	SafeThreadQuit(m_pThread_FT, hThread_FT, m_flags.ftRunning);				// FT 센서 쓰레드 종료
	SafeThreadQuit(m_pThread_FC, hThread_FC, m_flags.robotRunning);				// 로봇 구동 쓰레드 종료
	SafeThreadQuit(m_pThread_Air, hThread_Air, m_flags.airRunning);				// 공압 제어 쓰레드 종료

	m_logCv.notify_one();														// 로그 쓰레드 종료를 위한 조건 변수 알림
	SafeThreadQuit(m_pThread_Logger, hThread_Logger, m_flags.logThreadRunning); // 로그 쓰레드 종료
	KillTimer(1);																// GUI 주기 타이머 일괄 해제타이머 종료
}

/////////////////////////////////////////////
// 4. 쓰레드 실행 함수
/////////////////////////////////////////////

// =========================================
// FT 센서 쓰레드 함수
UINT CRobotCommSWDJv5Dlg::Thread_FT_Receive(LPVOID pParam)
{
	auto* g_pDlg = static_cast<CRobotCommSWDJv5Dlg*>(pParam);

	// FT 센서 쓰레드의 우선 순위 설정
	HANDLE hThread = GetCurrentThread();										// 현재 쓰레드 핸들 가져오기
	SetThreadPriority(hThread, THREAD_PRIORITY_BELOW_NORMAL);
	SetThreadAffinityMask(hThread, 1ULL << 1);									// CPU #1번에 고정

	// 고분해능 타이머 초기화
	// QueryPerformanceFrequency => 1초당 tick 수
	LARGE_INTEGER qpf;
	QueryPerformanceFrequency(&qpf);
	const LONGLONG one_ms_tick = qpf.QuadPart / 1000LL;

	// QueryPerformanceCounter => 시작 시각
	LARGE_INTEGER startTick;
	QueryPerformanceCounter(&startTick);

	// nextTick: 첫 목표 시각 = startTick + one_ms_tick
	LARGE_INTEGER nextTick = startTick;
	nextTick.QuadPart += one_ms_tick;										// 첫 주기 목표

	// 반복 주파수 측정용 변수(1초마다 실제 주기 계산)
	LARGE_INTEGER freqStart = startTick;
	int i_freq = 0;

	UNREFERENCED_PARAMETER(pParam);

	if (!g_pDlg)
		return 1;

	// Low-pass 필터 설정 및 생성
	const double fs = 1000.0;           // 실측 샘플링 주파수
	const double cutoffHz = 5.0;        // 차단 주파수

	// 접촉력 및 토크에 대한 Low-pass 필터 생성
	LowPassFilter lpfFx(fs, cutoffHz);
	LowPassFilter lpfFy(fs, cutoffHz);
	LowPassFilter lpfFz(fs, cutoffHz);

	LowPassFilter lpfTx(fs, cutoffHz);
	LowPassFilter lpfTy(fs, cutoffHz);
	LowPassFilter lpfTz(fs, cutoffHz);

	// FT 센서 데이터 수신 루프 시작
	while (g_pDlg->m_flags.ftRunning.load())
	{
		auto ts = system_clock::now();

		// 1) FT 센서 데이터 수신 및 해석 (내부적으로 상태 업데이트)
		g_pDlg->m_ftSensor.ProcessMessage(100);

		// 2) 바이어스 적용 힘/토크 얻기
		auto biasedForce = g_pDlg->m_ftSensor.getState().getBiasedForce();
		auto biasedTorque = g_pDlg->m_ftSensor.getState().getBiasedTorque();

		// 3) 힘/토크를 로봇 기준 좌표계로 변환
		auto robotPose = g_pDlg->m_robotState.getSnapshot().tcpPos;
		Transformation_Force tfF(biasedForce.data(), robotPose.data());
		std::array<float, 3> forceBase;
		tfF.transformForce(forceBase.data());
		forceBase[0] *= -1.0f;
		forceBase[2] *= -1.0f;

		Transformation_Torque tfT(biasedTorque.data(), robotPose.data());
		std::array<float, 3> torqueBase;
		tfT.transformTorque(torqueBase.data());
		torqueBase[0] *= -1.0f;
		torqueBase[2] *= -1.0f;

		// 4) 힘/토크 센서 데이터의 Lowpass filter 적용
		std::array<float, 3> filteredForce = {
			static_cast<float>(lpfFx.filter(forceBase[0])),
			static_cast<float>(lpfFy.filter(forceBase[1])),
			static_cast<float>(lpfFz.filter(forceBase[2]))
		};
		std::array<float, 3> filteredTorque = {
			static_cast<float>(lpfTx.filter(torqueBase[0])),
			static_cast<float>(lpfTy.filter(torqueBase[1])),
			static_cast<float>(lpfTz.filter(torqueBase[2]))
		};

		// 5) FTState에 계산 결과 모두 기록
		g_pDlg->m_ftSensor.getState().setAll_FT_cal(
			biasedForce, biasedTorque,
			forceBase, torqueBase,
			filteredForce, filteredTorque
			);
		g_pDlg->m_ftSensor.getState().updateSnapshot();

		//////////////////////////////////////////////////////////////////////////
		////while문 속도 제어+++++++++++++++++++++++++++++++++++++++++++++++++++++
		//////////////////////////////////////////////////////////////////////////
		// 1ms 경과까지 스핀 대기
		for (;;)
		{
			LARGE_INTEGER now;
			QueryPerformanceCounter(&now);					// 현재 시각
			if (now.QuadPart >= nextTick.QuadPart) break;	// 1ms 경과
			_mm_pause();									// 20~30 ns 휴식
		}

		// 다음 루프를 위해 nextTick 업데이트
		nextTick.QuadPart += one_ms_tick;

		//////////////////////////////////////////////////////////////////////////
		////주기 측정 및 출력+++++++++++++++++++++++++++++++++++++++++++++++++++++
		//////////////////////////////////////////////////////////////////////////
		i_freq++;
		if (i_freq == g_pDlg->m_setting.set_Hz)
		{
			LARGE_INTEGER now;
			QueryPerformanceCounter(&now);
			double elapsed_s = (double(now.QuadPart - freqStart.QuadPart) / double(qpf.QuadPart));
			g_pDlg->var_freq_ft_sensor = 1.0 / (elapsed_s / static_cast<double>(g_pDlg->m_setting.set_Hz));
			i_freq = 0;
			freqStart = now;
		}
		//////////////////////////////////////////////////////////////////////////
	}

	g_pDlg->m_flags.ftRunning.store(false);

	return 0;
}

// =========================================
// 서보 제어 쓰레드 함수
UINT CRobotCommSWDJv5Dlg::Thread_Servo(LPVOID pParam)
{
	CRobotCommSWDJv5Dlg* g_pDlg = (CRobotCommSWDJv5Dlg*)pParam;

	// 서보제어 쓰레드 우선순위 설정
	HANDLE hThread = GetCurrentThread();
	SetThreadPriority(hThread, THREAD_PRIORITY_TIME_CRITICAL);
	SetThreadAffinityMask(hThread, 1ULL << 0);								// CPU #0번에 고정

	g_pDlg->m_servoctrl.jacc = { 20.0, 20.0 };

	// 고분해능 타이머 초기화
	// QueryPerformanceFrequency => 1초당 tick 수
	LARGE_INTEGER qpf;
	QueryPerformanceFrequency(&qpf);
	const LONGLONG one_ms_tick = qpf.QuadPart / 1000LL;

	// QueryPerformanceCounter => 시작 시각
	LARGE_INTEGER startTick;
	QueryPerformanceCounter(&startTick);

	// nextTick: 첫 목표 시각 = startTick + one_ms_tick
	LARGE_INTEGER nextTick = startTick;
	nextTick.QuadPart += one_ms_tick;										// 첫 주기 목표

	// 반복 주파수 측정용 변수(1초마다 실제 주기 계산)
	LARGE_INTEGER freqStart = startTick;
	int i_freq = 0;

	// 안전 범위 위반 여부 확인을 위한 헬퍼 함수
	auto stopRobotWithError = [&](const CString& errorMsg) 
		{
		Drfl.stop_rt_control();												// RT 파이프 정리
		Drfl.set_safety_mode(SAFETY_MODE_AUTONOMOUS,
			SAFETY_MODE_EVENT_STOP);										// SC 정지 명령
		Drfl.set_robot_control(CONTROL_RESET_SAFET_OFF);					// 서보 오프
		g_pDlg->m_flags.realtimeConnected.store(false);
		g_pDlg->m_flags.servoRunning.store(false);
		g_pDlg->m_flags.ftRunning.store(false);
		g_pDlg->m_flags.robotRunning.store(false);
		Status_gui_str = errorMsg;
		g_pDlg->var_status_gui.SetWindowTextW(Status_gui_str);
		Drfl.disconnect_rt_control();
		return true;
		};

	while (g_pDlg->m_flags.servoRunning.load())
	{
		// "속도 및 가속도 명령”준비 전에 현재 RT 연결 상태 검사
		if (!g_pDlg->m_flags.realtimeConnected.load())
		{
			if (g_pDlg->GetSafeHwnd())
				::PostMessage(g_pDlg->GetSafeHwnd(), WM_USER + 100, 0, 0);
		}

		// RT 데이터 읽기 
		//	- 현재 쓰레드에서 외부 모든 쓰레드에 활용되는 로봇 정보 데이터를 수집함
		//	- 외부 쓰레드에서 로봇 정보를 활용 시 "getSnapshot" 함수를 활용
		auto pRt = Drfl.read_data_rt();
		if (pRt) {
			float desVel[3] = { g_pDlg->m_servoctrl.vx_cmd, g_pDlg->m_servoctrl.vy_cmd, g_pDlg->m_servoctrl.vz_cmd };
			g_pDlg->m_robotState.updateAllRawData(
				pRt->actual_tcp_position,
				pRt->actual_flange_position,
				pRt->actual_joint_position,
				pRt->actual_tcp_velocity,
				pRt->actual_flange_velocity,
				pRt->actual_joint_velocity,
				desVel,
				6
			);
			g_pDlg->m_robotState.updateSnapshot();
		}
		
		// 로봇 & 센서 정보 수집
		auto Th_robotData_servo = g_pDlg->m_robotState.getSnapshot();								// 로봇 정보 수신
		auto Th_sensorData_servo = g_pDlg->m_ftSensor.getSnapshot();								// FT 센서 수신

		// 안전 범위 검사 (포지션, 토크 등) => 문제가 있으면 즉시 로봇 구동을 중단하고 모든 쓰레드 종료
		//  - 사용자의 로봇 사용 조건에 맞춰 수정 필요
		if ((Th_robotData_servo.flangePos[0] < -360.0f) || (Th_robotData_servo.flangePos[0] > 480.0f)
			|| (Th_robotData_servo.flangePos[1] < 280.0f) || (Th_robotData_servo.flangePos[1] > 456.0f)
			|| (Th_robotData_servo.flangePos[2] < 330.0f) || (Th_robotData_servo.flangePos[2] > 500.0f))
		{
			if (stopRobotWithError(_T("엔드이펙터 위치 범위 초과로 인한 정지")))
				break;
		}
		else if ((Th_robotData_servo.jointAng[0] < 30.0) || (Th_robotData_servo.jointAng[0] > 135.0)
			|| (Th_robotData_servo.jointAng[1] < -15.0) || (Th_robotData_servo.jointAng[1] > 80.0)
			|| (Th_robotData_servo.jointAng[2] < 10.0) || (Th_robotData_servo.jointAng[2] > 110.0)
			|| (Th_robotData_servo.jointAng[3] < -30.0) || (Th_robotData_servo.jointAng[3] > 30.0)
			|| (Th_robotData_servo.jointAng[4] < 60.0) || (Th_robotData_servo.jointAng[4] > 130.0)
			|| (Th_robotData_servo.jointAng[5] < -135.0) || (Th_robotData_servo.jointAng[5] > -35.0))
		{
			if (stopRobotWithError(_T("관절 각도 범위 초과로 인한 정지")))
				break;
		}
		else if (abs(Th_sensorData_servo.filteredForce[0]) > 30.0)
		{
			if (stopRobotWithError(_T("Fx 접촉력 초과로 인한 정지")))
				break;
		}
		else if (abs(Th_sensorData_servo.filteredForce[1]) > 30.0)
		{
			if (stopRobotWithError(_T("Fy 접촉력 초과로 인한 정지")))
				break;
		}
		else if (abs(Th_sensorData_servo.filteredForce[2]) > 100.0)
		{
			//if (stopRobotWithError(_T("Fz 접촉력 초과로 인한 정지")));
				//break;
			printf("Fz 접촉력 초과로 인한 정지!\n");

		}
		else if (abs(Th_robotData_servo.tcpVel[2]) > 30.0)
		{
			if (stopRobotWithError(_T("z방향 속도 초과로 인한 정지")))
				break;
		}
		else
		{
			if (g_pDlg->m_flags.loadTraj.load() == false)
			{
				if (g_pDlg->m_setting.Control_Step != 2)	// 공압 제어 시 로봇 동작 명령 Off
				{
					// ===================================
				// 평면 경로 구동용 (speedl 호출)
				// 평면 경로 구동 시 원하는 이동 속도값 갱신
					g_pDlg->m_servoctrl.servo_flange_vel_des_key[0] = g_pDlg->m_servoctrl.vx_cmd.load();
					g_pDlg->m_servoctrl.servo_flange_vel_des_key[1] = g_pDlg->m_servoctrl.vy_cmd.load();
					g_pDlg->m_servoctrl.servo_flange_vel_des_key[2] = g_pDlg->m_servoctrl.vz_cmd.load();

					// 실제 로봇 제어기로 로봇 구동 명령 송신
					int servo_msg = Drfl.speedl(
						g_pDlg->m_servoctrl.servo_flange_vel_des_key.data(),
						g_pDlg->m_servoctrl.jacc.data(),
						0.001f
					);

					// 실제 로봇 제어기로 로봇 구동 명령 송신 실패시 알림 및 로봇 구동 종료
					if (servo_msg != 1)
					{
						g_pDlg->HandleServoError(servo_msg, _T("평면"));
						break;
					}
					//printf("[INFO] 서보 명령 전송 중\n");
				}
				else
				{
					//printf("[INFO] 서보 명령 전송 종료!!!\n");
				}
			}
			else
			{
				// ===================================
				// 곡면 경로 구동용 (speedl 호출)
				// 실제 로봇 제어기로 로봇 구동 명령 송신

				g_pDlg->m_servoctrl.servo_tcp_vel_des_key = {
					g_pDlg->m_servoctrl.vx_cmd.load(),
					g_pDlg->m_servoctrl.vy_cmd.load(),
					g_pDlg->m_servoctrl.vz_cmd.load(),
					g_pDlg->m_servoctrl.roll_cmd.load(),
					g_pDlg->m_servoctrl.pitch_cmd.load(),
					g_pDlg->m_servoctrl.yaw_cmd.load()
				};

				g_pDlg->m_servoctrl.servo_tcp_acc_des_key = {
					100.0f, 100.0f
				};

				int servo_msg = Drfl.speedl(
					g_pDlg->m_servoctrl.servo_tcp_vel_des_key.data(),
					g_pDlg->m_servoctrl.servo_tcp_acc_des_key.data(),
					0.001f
				);
				// 실제 로봇 제어기로 로봇 구동 명령 송신 실패시 알림 및 로봇 구동 종료
				if (servo_msg != 1)
				{
					g_pDlg->HandleServoError(servo_msg, _T("곡면"));
					break;
				}
			}
			
		}
		//////////////////////////////////////////////////////////////////////////
		////while문 속도 제어+++++++++++++++++++++++++++++++++++++++++++++++++++++
		//////////////////////////////////////////////////////////////////////////
		// 1ms 경과까지 스핀 대기
		for (;;)
		{
			LARGE_INTEGER now;
			QueryPerformanceCounter(&now);					// 현재 시각
			if (now.QuadPart >= nextTick.QuadPart) break;	// 1ms 경과
			_mm_pause();									// 20~30 ns 휴식
		}

		// 다음 루프를 위해 nextTick 업데이트
		nextTick.QuadPart += one_ms_tick;

		//////////////////////////////////////////////////////////////////////////
		////주기 측정 및 출력+++++++++++++++++++++++++++++++++++++++++++++++++++++
		//////////////////////////////////////////////////////////////////////////
		i_freq++;
		if (i_freq == g_pDlg->m_setting.set_Hz_servo)
		{
			LARGE_INTEGER now;
			QueryPerformanceCounter(&now);
			double elapsed_s = (double(now.QuadPart - freqStart.QuadPart) / double(qpf.QuadPart));
			g_pDlg->var_freq_servo = 1.0 / (elapsed_s / static_cast<double>(g_pDlg->m_setting.set_Hz));
			i_freq = 0;
			freqStart = now;
		}
	}

	g_pDlg->m_flags.servoRunning.store(false);
	return 0;
}

// =========================================
// 공압 제어 쓰레드 함수
UINT CRobotCommSWDJv5Dlg::Thread_AirControl(LPVOID pParam)
{
	//auto* g_pDlg = static_cast<CRobotCommSWDJv5Dlg*>(pParam);
	CRobotCommSWDJv5Dlg* g_pDlg = (CRobotCommSWDJv5Dlg*)pParam;

	// 공압 제어 쓰레드 우선순위 설정
	HANDLE hThread = GetCurrentThread();
	SetThreadPriority(hThread, THREAD_PRIORITY_BELOW_NORMAL);
	SetThreadAffinityMask(hThread, 1ULL << 3);							// CPU #3번(core 3) 고정
	
	auto& ac = g_pDlg->m_airctrl;

	// 고분해능 타이머 초기화
	// QueryPerformanceFrequency => 1초당 tick 수
	LARGE_INTEGER qpf;
	QueryPerformanceFrequency(&qpf);
	const LONGLONG one_ms_tick = qpf.QuadPart / 1000LL;

	// QueryPerformanceCounter => 시작 시각
	LARGE_INTEGER startTick;
	QueryPerformanceCounter(&startTick);

	// nextTick: 첫 목표 시각 = startTick + one_ms_tick
	LARGE_INTEGER nextTick = startTick;
	nextTick.QuadPart += one_ms_tick;										// 첫 주기 목표

	// 공압 출력 루프 시작
	while (g_pDlg->m_flags.airRunning.load())
	{
		ac.updateOutputs();												// 최종 출력할 공압값 업데이트 함수
		//ac.updateInputs();												// 공압 시스템 입력값 업데이트 함수

		//////////////////////////////////////////////////////////////////////////
		////while문 속도 제어+++++++++++++++++++++++++++++++++++++++++++++++++++++
		//////////////////////////////////////////////////////////////////////////
		// 1ms 경과까지 스핀 대기
		for (;;)
		{
			LARGE_INTEGER now;
			QueryPerformanceCounter(&now);					// 현재 시각
			if (now.QuadPart >= nextTick.QuadPart) break;	// 1ms 경과
			_mm_pause();									// 20~30 ns 휴식
		}

		// 다음 루프를 위해 nextTick 업데이트
		nextTick.QuadPart += one_ms_tick;
	}

	g_pDlg->m_flags.airRunning.store(false);

	return 0;
}

// =========================================
// 로봇 구동 쓰레드 함수 (평면 구동)
UINT CRobotCommSWDJv5Dlg::Thread_Contact_Flat(LPVOID pParam)
{
	CRobotCommSWDJv5Dlg* g_pDlg = (CRobotCommSWDJv5Dlg*)pParam;
	
	// 로봇 구동 쓰레드 우선순위 설정
	HANDLE hThread = GetCurrentThread();
	SetThreadPriority(hThread, THREAD_PRIORITY_HIGHEST);
	SetThreadAffinityMask(hThread, 1ULL << 2);					// CPU #2번(core 2) 고정

	// 데이터 저장용 시간 변수
	system_clock::time_point t_start, t_cd;
	nanoseconds t_stamp_ns, t_stamp_cd_ns;
	float t_stamp_ns_float = 0.0;
	float t_stamp_cd_ns_float = 0.0;
	
	// ===============================================
	// 고분해능 타이머 초기화
	// QueryPerformanceFrequency → 1초당 tick 수
	LARGE_INTEGER qpf;
	QueryPerformanceFrequency(&qpf);
	const LONGLONG one_ms_tick = qpf.QuadPart / 1000LL;

	// QueryPerformanceCounter → 시작 시각
	LARGE_INTEGER startTick;
	QueryPerformanceCounter(&startTick);		// 시작 시각

	// nextTick: 첫 목표 시각 = startTick + one_ms_tick
	LARGE_INTEGER nextTick = startTick;
	nextTick.QuadPart += one_ms_tick;		// 첫 주기 목표

	// 반복 주파수 측정용 변수(1초마다 실제 주기 계산)
	LARGE_INTEGER freqStart = startTick;
	int i_freq = 0;

	// ===============================================
	// 실험 설정
	g_pDlg->m_setting.Target_Force_N.store(-30.0f);				// 목표 접촉력 설정 (N)   [음수로 설정해줘야 로봇 베이스 좌표계와 동일한 방향]
	g_pDlg->m_setting.Force_limit_N.store(50.0f);				// 접촉력 제한값 설정 (N)
	g_pDlg->m_setting.Target_vz.store(5.0f);					// 목표 접촉 속도 설정 (mm/s) [양수로 설정해줘야 로봇 베이스 좌표계와 동일한 방향]
	int Saturation_time = 5000;								// 접촉 유지 시간 설정 (ms)
	g_pDlg->m_setting.First_Contact.store(true);
	
	// ==============================================
	// 로봇 동작 시 초기 로봇의 시작 및 종료에 대한 기준 위치값 측정
	auto Init_flange_pos_load = g_pDlg->m_robotState.getSnapshot();
	g_pDlg->m_robotState.saveInitialflangePosition(Init_flange_pos_load.flangePos.data(), 6);
	auto initialPosArray_flange = g_pDlg->m_robotState.getInitialflangePositionArray();

	// ===============================================
	// Cubic smoothstep 함수로 생성되는 속도 경로로 로봇을 하강 & 상승 구동하기 위한 설정
	static ProfileConfig approachConfig;
	static ProfileConfig retractConfig;
	static float start_pos_z = 0.0f;
	static auto rampStartTime = std::chrono::system_clock::now();

	// ===============================================
	// 공압 관련 변수 초기화
	const double pressureRampDownDuration_sec = 3.0;			// 압력 감소 시간 (초)
	static auto pressureRampStartTime = std::chrono::system_clock::now();
	static float initialChamberPressure = 0.0f;

	// ===============================================
	// PID 제어기 관련 변수 초기화
	const double base_pressure_mpa = 0.2;						// MFT 초기 기본 압력 (MPa) => 최대 0.4MPa까지 가압 가능
	const double MFT_diameter_mm = 94.9;						// MFT 직경 (mm)
	double MFT_radius_m = MFT_diameter_mm / 2.0 * 1e-3;			// MFT 반지름 (m)					
	double A_m2 = M_PI * MFT_radius_m * MFT_radius_m;			// MFT 단면적 (m^2)

	// PID 게인 및 바운드 설정
	g_pDlg->m_pidctrl.setGains(
		10.0,			// Kp
		1.0,			// Ki
		0.0);			// Kd
	g_pDlg->m_pidctrl.setOutputLimits(
		-1500.0,		// min
		1500.0);		// max

	// ===============================================
	// 데이터 기록 시작
	g_pDlg->m_flags.logThreadRunning.store(true);
	g_pDlg->m_pThread_Logger = AfxBeginThread(Thread_Logger, g_pDlg);

	t_start = system_clock::now();

	// 로봇 구동 루프 시작 (평면 경로)
	while (g_pDlg->m_flags.robotRunning.load())
	{
		auto ts = std::chrono::steady_clock::now();

		// 로봇 & 센서 정보 수신
		auto Th_robotData_flat = g_pDlg->m_robotState.getSnapshot();
		auto Th_sensorData_flat = g_pDlg->m_ftSensor.getSnapshot();

		// FT 센서 연결 여부 확인
		if (g_pDlg->m_flags.ftRunning.load())
		{	
			// 평면 구동 과정에서 일시 중지 기능 (GUI에서 구동 종료 버튼)
			//  - Control Step = 3의 조건을 수행하여 현재 구동을 멈추고 로봇 구동 마무리 동작을 수행
			if (g_pDlg->m_flags.flat_stop.load() == true)
			{
				g_pDlg->m_setting.vx_mms.store(0.0);
				g_pDlg->m_setting.vy_mms.store(0.0);
				g_pDlg->m_setting.vz_mms.store(0.0);

				g_pDlg->m_servoctrl.vx_cmd.store(g_pDlg->m_setting.vx_mms.load());
				g_pDlg->m_servoctrl.vy_cmd.store(g_pDlg->m_setting.vy_mms.load());
				g_pDlg->m_servoctrl.vy_cmd.store(g_pDlg->m_setting.vz_mms.load());

				g_pDlg->m_setting.Control_Step = 3;
				g_pDlg->m_setting.First_Contact.store(true);
				g_pDlg->m_flags.flat_stop.store(false);
			}

			// Control Step.0: 툴을 금형 시편에 접촉하기 위한 하강 동작
			if (g_pDlg->m_setting.Control_Step == 0)	
			{
				if (g_pDlg->m_setting.First_Contact.load() == true) {
					// 1. 상태 초기화
					rampStartTime = std::chrono::system_clock::now();
					g_pDlg->m_setting.First_Contact.store(false);

					// 2. 하강 시작 위치 기록
					start_pos_z = Th_robotData_flat.flangePos[2];

					// 3. 하강 프로파일 설정
					approachConfig.direction = -1;
					approachConfig.max_velocity = g_pDlg->m_setting.Target_vz;
					approachConfig.final_velocity = 1.0f;
					approachConfig.target_z = 385.0f;
					approachConfig.ramp_duration_sec = 3.0; // 가속 시간 설정
					approachConfig.move_distance = std::abs(approachConfig.target_z - start_pos_z);	// 총 이동 거리 계산
				}
				auto now = std::chrono::system_clock::now();
				double elapsed_sec = std::chrono::duration<double>(now - rampStartTime).count();
				float current_pos_z = Th_robotData_flat.flangePos[2];

				// 설정된 approachConfig를 사용하여 속도 계산
				float final_target_vz = g_pDlg->m_velProfile.calculate(approachConfig, elapsed_sec, current_pos_z, start_pos_z);
				g_pDlg->m_setting.vz_mms.store(final_target_vz);

				// 목표 접촉력 도달 시 다음 단계로 이동
				if (Th_sensorData_flat.filteredForce[2] <= g_pDlg->m_setting.Target_Force_N)
				{
					g_pDlg->m_setting.Control_Step = 1;
					g_pDlg->m_setting.vz_mms.store(0.0f);
					g_pDlg->m_setting.First_Contact.store(true);
				}

				g_pDlg->m_servoctrl.vz_cmd.store(g_pDlg->m_setting.vz_mms);	// [mm/s]

				Status_gui_str.Format(_T("[평면 구동] Control Step 0:로봇 하강 중... (Vz: %.2f)"), -g_pDlg->m_setting.vz_mms.load());
				g_pDlg->var_status_gui.SetWindowTextW(Status_gui_str);
			}
			// Control Step.1: 일정 시간 동안 접촉 유지 (사용자가 정한 시간 동안)
			else if (g_pDlg->m_setting.Control_Step == 1)
			{
				if (g_pDlg->m_setting.First_Contact.load() == true)
				{
					t_cd = system_clock::now();
					g_pDlg->m_setting.First_Contact.store(false);
				}
				else
				{
					t_stamp_cd_ns = system_clock::now() - t_cd;
					t_stamp_cd_ns_float = float(t_stamp_cd_ns.count());
					float t_stamp_cd_ms_float = t_stamp_cd_ns_float / 1000000.0f;

					// 목표 접촉 유지 시간 초과시 접촉 유지 정지 후 다음 단계로 이동
					if (t_stamp_cd_ms_float > Saturation_time)
					{
						g_pDlg->m_setting.vz_mms.store(0.0f);		// [mm/s] 접촉 유지 정지
						g_pDlg->m_setting.First_Contact.store(true);
						g_pDlg->m_setting.Control_Step = 2;
					}
					g_pDlg->m_setting.Contact_time = static_cast<int>(Saturation_time - t_stamp_cd_ms_float) * 0.001f;
				}

				float err_f = g_pDlg->m_setting.Target_Force_N.load() - Th_sensorData_flat.filteredForce[2];	// [N] 접촉력 오차
				float kp = 0.05f;

				g_pDlg->m_setting.vz_mms.store(kp * err_f);		// [mm/s] 접촉력 오차 기반 z방향 속도 설정
				g_pDlg->m_servoctrl.vz_cmd.store(g_pDlg->m_setting.vz_mms);	// [mm/s]

				Status_gui_str.Format(_T("[평면 구동] Control Step 1: 로봇 접촉 수렴중..."));
				g_pDlg->var_status_gui.SetWindowTextW(Status_gui_str);
			}
			// Control Step.2: PID 제어기를 적용한 공압 제어 및 평면 구동 시작
			else if (g_pDlg->m_setting.Control_Step == 2)
			{
				if (g_pDlg->m_setting.First_Contact.load() == true)
				{
					g_pDlg->m_setting.First_Contact.store(false);
					g_pDlg->m_setting.vx_mms.store(5.0);					// X축에 대한 이동 방향 & 속도 설정

					g_pDlg->m_airctrl.setDesiredSpindlePressure(0.0);

					// PID 컨트롤러 리셋
					g_pDlg->m_pidctrl.reset();

					// Ki 게인이 0이 아니어야 Integral 항 설정이 의미가 있습니다.
					// Ki가 0이라면 이 방법은 효과가 없으며, 다른 방법을 사용해야 합니다.
					if (g_pDlg->m_pidctrl.getKi() > 0) {
						// 저장해둔 초기 PID 출력값을 Ki로 나누어 초기 Integral 항 값을 계산합니다.
						// (PID 공식에서 output에 Integral 기여분은 ki * integral 이므로)
						double initial_integral = g_pDlg->m_pidInitialOutput_N / g_pDlg->m_pidctrl.getKi();

						// 새로 만든 함수를 이용해 Integral 항을 설정합니다.
						g_pDlg->m_pidctrl.setIntegral(initial_integral);
					}

					g_pDlg->m_setting.Target_Force_N.store(-30.0f);			// 목표 접촉력 변경 [N]

					Status_gui_str.Format(_T("[평면 구동] Control Step 2: 평면 구동 & PID 힘 제어 시작"));
					g_pDlg->var_status_gui.SetWindowTextW(Status_gui_str);
				}
				else
				{
					static auto last_time = std::chrono::steady_clock::now();
					static bool is_first_run = true;

					auto current_time = std::chrono::steady_clock::now();
					double actual_dt = std::chrono::duration<double>(current_time - last_time).count();
					last_time = current_time;

					// PID 제어 주기 설정
					if (is_first_run)
					{
						actual_dt = 0.001;
						is_first_run = false;
					}
					else if (actual_dt > 0.01)
					{
						actual_dt = 0.01;
					}

					double setpoint_force = abs(g_pDlg->m_setting.Target_Force_N.load());					// [N] 목표 접촉력 (양수로 설정)
					double measured_force = abs(Th_sensorData_flat.filteredForce[2]);						// [N] 측정된 접촉력 (양수로 설정)

					// PID 제어기 계산
					PIDController::Result result = g_pDlg->m_pidctrl.calculate(setpoint_force, measured_force, actual_dt);

					double force_correction_N = result.output;												// [N] PID 제어기 오차 기반 접촉력 보정값 (PID 출력값)

					// 접촉력 => 공압 변환
					double pressure_correction_mpa = force_correction_N / A_m2 * 1e-6;						// [MPa] 공압 보정값 (N -> MPa 변환)
					double new_target_pressure_mpa = base_pressure_mpa + pressure_correction_mpa;			// [MPa] 새로운 목표 공압값 (기본 압력 + 보정값)
					new_target_pressure_mpa = std::clamp(new_target_pressure_mpa, 0.0, 0.4);				// [MPa] 공압 제한 (0.0 ~ 0.4 MPa)

					g_pDlg->m_airctrl.setDesiredChamberPressure(new_target_pressure_mpa);					// [MPa] 출력 챔버 공압 설정

					// 현재 로봇위치가 평면 구동 종료 위치를 넘어가는지 확인
					//   - 평면 구동 종료 위치는 사용자가 설정한 x_pos_bound[1] 값
					//   - 평면 구동 종료 위치에 도달하면 Control Step = 3 단계로 이동
					if (Th_robotData_flat.flangePos[0] >= g_pDlg->m_setting.x_pos_bound[1])
					{
						g_pDlg->m_setting.vx_mms.store(0.0);	// [mm/s]
						g_pDlg->m_setting.First_Contact.store(true);
						g_pDlg->m_setting.Control_Step = 3;
					}
				}

				g_pDlg->m_servoctrl.vx_cmd.store(g_pDlg->m_setting.vx_mms);									// [mm/s]
				g_pDlg->m_servoctrl.vz_cmd.store(g_pDlg->m_setting.vz_mms);									// [mm/s]
				g_pDlg->m_airctrl.setDesiredChamberPressure(g_pDlg->m_airctrl.desiredChamberPressure());	// [MPa] 출력 챔버 공압 최종 설정
			}
			// Control Step.3: 평면 경로 구동 마무리
			else if (g_pDlg->m_setting.Control_Step == 3)
			{
				if (g_pDlg->m_setting.First_Contact.load() == true)
				{
					// 1. 상태 초기화
					rampStartTime = system_clock::now();
					g_pDlg->m_setting.First_Contact.store(false);

					// 2. 동작 시작 시점의 상태 저장
					start_pos_z = Th_robotData_flat.flangePos[2];

					// 3. 상승(Retract) 프로파일 설정
					retractConfig.direction = 1;								// 상승
					retractConfig.max_velocity = g_pDlg->m_setting.Target_vz;	// 상승 속도
					retractConfig.final_velocity = 0.0f;						// 목표 지점에서 정지
					retractConfig.target_z = initialPosArray_flange[2];				// GUI에서 설정한 초기 Z 위치로 설정
					retractConfig.ramp_duration_sec = 3;						// 상승 가/감속 시간
					retractConfig.move_distance = std::abs(retractConfig.target_z - start_pos_z); // 총 이동 거리 계산

					// 4. 스핀들 공압 즉시 OFF
					g_pDlg->m_airctrl.setDesiredChamberPressure(0.0);			// 챔버 공압 OFF
					g_pDlg->m_airctrl.setDesiredSpindlePressure(0.0);			// 스핀들 공압 OFF

					Status_gui_str.Format(_T("[평면 구동] Control Step 3: 공압 제어 완료 및 초기 위치로 이동"));
					g_pDlg->var_status_gui.SetWindowTextW(Status_gui_str);
				}
				else
				{
					auto now = std::chrono::system_clock::now();
					float current_pos_z = Th_robotData_flat.flangePos[2];

					// 프로파일 기반 로봇 상승 로직
					double motion_elapsed_sec = std::chrono::duration<double>(now - rampStartTime).count();
					float final_target_vz = g_pDlg->m_velProfile.calculate(retractConfig, motion_elapsed_sec, current_pos_z, start_pos_z);
					g_pDlg->m_setting.vz_mms.store(final_target_vz);

					// 초기 로봇 시작 위치 이상으로 높은 위치에 TCP가 도달하면 구동 종료
					//if (initialPosArray_flange[2] <= Th_robotData_flat.flangePos[2])
					if (422.5 <= Th_robotData_flat.flangePos[2])
					{
						g_pDlg->m_setting.vz_mms = 0.0;
						Status_gui_str.Format(_T("[평면 구동] Control Step 3: 로봇 구동 종료"));
						g_pDlg->var_status_gui.SetWindowTextW(Status_gui_str);

						g_pDlg->OnBnClickedButRobotDisconnect();
					}
				}
				g_pDlg->m_servoctrl.vz_cmd.store(g_pDlg->m_setting.vz_mms);	// [mm/s]
				Status_gui_str.Format(_T("[평면 구동] Control Step 3: 프로파일 상승 중... (Vz: %.2f)"), g_pDlg->m_setting.vz_mms.load());
				g_pDlg->var_status_gui.SetWindowTextW(Status_gui_str);
			}
		}
		//////////////////////////////////////////////////////////////////////////
		////데이터 기록+++++++++++++++++++++++++++++++++++++++++++++++++++++
		//////////////////////////////////////////////////////////////////////////		
		t_stamp_ns = system_clock::now() - t_start;
		t_stamp_ns_float = float(t_stamp_ns.count());
		float t_stamp_ms_float = t_stamp_ns_float / 1000000.0f;

		LogData log;
		log.data.reserve(41); // 로그 데이터 크기 예약

		log.data.push_back(t_stamp_ms_float);
		log.data.insert(log.data.end(), Th_robotData_flat.flangePos.begin(), Th_robotData_flat.flangePos.end());
		log.data.insert(log.data.end(), Th_robotData_flat.jointAng.begin(), Th_robotData_flat.jointAng.end());
		log.data.insert(log.data.end(), Th_robotData_flat.tcpVel.begin(), Th_robotData_flat.tcpVel.end());
		log.data.insert(log.data.end(), Th_robotData_flat.jointVel.begin(), Th_robotData_flat.jointVel.end());
		log.data.insert(log.data.end(), Th_sensorData_flat.filteredForce.begin(), Th_sensorData_flat.filteredForce.end());
		log.data.insert(log.data.end(), Th_sensorData_flat.filteredTorque.begin(), Th_sensorData_flat.filteredTorque.end());

		log.data.push_back(g_pDlg->m_setting.vz_mms);
		log.data.push_back((float)g_pDlg->m_pidctrl.getLastError());
		log.data.push_back((float)g_pDlg->m_airctrl.sendChamberPressure());
		log.data.push_back((float)g_pDlg->m_airctrl.sendChamberVoltage());
		log.data.push_back((float)g_pDlg->m_airctrl.sendSpindlePressure());
		log.data.push_back((float)g_pDlg->m_airctrl.sendSpindleVoltage());
		log.data.push_back((float)g_pDlg->m_airctrl.feedbackChamberPressure());
		log.data.push_back((float)g_pDlg->m_airctrl.feedbackChamberVoltage());
		log.data.push_back((float)g_pDlg->m_airctrl.feedbackSpindlePressure());
		log.data.push_back((float)g_pDlg->m_airctrl.feedbackSpindleVoltage());
		{
			std::lock_guard<std::mutex> lock(g_pDlg->m_logMutex);
			g_pDlg->m_logQueue.push(log);
		}
		g_pDlg->m_logCv.notify_one();

		////////////////////////////////////////////////////////////////////////////
		//////while문 속도 제어+++++++++++++++++++++++++++++++++++++++++++++++++++++
		////////////////////////////////////////////////////////////////////////////
		// 1ms 경과까지 스핀 대기
		for (;;)
		{
			LARGE_INTEGER now;
			QueryPerformanceCounter(&now);					// 현재 시각
			if (now.QuadPart >= nextTick.QuadPart) break;	// 1ms 경과
			_mm_pause();									// 20~30 ns 휴식
		}

		// 다음 루프를 위해 nextTick 업데이트
		nextTick.QuadPart += one_ms_tick;

		////////////////////////////////////////////////////////////////////////////
		//////주기 측정 및 출력+++++++++++++++++++++++++++++++++++++++++++++++++++++
		////////////////////////////////////////////////////////////////////////////
		i_freq++;
		if (i_freq == g_pDlg->m_setting.set_Hz)
		{
			LARGE_INTEGER now;
			QueryPerformanceCounter(&now);
			double elapsed_s = (double(now.QuadPart - freqStart.QuadPart) / double(qpf.QuadPart));
			g_pDlg->var_freq_main = 1.0 / (elapsed_s / static_cast<double>(g_pDlg->m_setting.set_Hz));
			i_freq = 0;
			freqStart = now;
		}
	}
	g_pDlg->m_flags.logThreadRunning.store(false);
	g_pDlg->m_flags.robotRunning.store(false);
	return 0;
}

// =========================================
// 로봇 구동 쓰레드 함수 (곡면 구동)
UINT CRobotCommSWDJv5Dlg::Thread_Contact_Curve(LPVOID pParam)
{
	CRobotCommSWDJv5Dlg* g_pDlg = (CRobotCommSWDJv5Dlg*)pParam;
	
	// 로봇 구동 쓰레드 우선순위 설정
	HANDLE hThread = GetCurrentThread();
	SetThreadPriority(hThread, THREAD_PRIORITY_HIGHEST);
	SetThreadAffinityMask(hThread, 1ULL << 2);					// CPU #2번(core 2) 고정

	// 데이터 저장용 시간 변수
	system_clock::time_point t_start, t_cd;
	nanoseconds t_stamp_ns;
	float t_stamp_ns_float = 0.0;

	// ===============================================
	// 고분해능 타이머 초기화
	// QueryPerformanceFrequency → 1초당 tick 수
	LARGE_INTEGER qpf;
	QueryPerformanceFrequency(&qpf);
	const LONGLONG one_ms_tick = qpf.QuadPart / 1000LL;

	// QueryPerformanceCounter → 시작 시각
	LARGE_INTEGER startTick;
	QueryPerformanceCounter(&startTick);						// 시작 시각

	// nextTick: 첫 목표 시각 = startTick + one_ms_tick
	LARGE_INTEGER nextTick = startTick;
	nextTick.QuadPart += one_ms_tick;							// 첫 주기 목표

	// 반복 주파수 측정용 변수(1초마다 실제 주기 계산)
	LARGE_INTEGER freqStart = startTick;
	int i_freq = 0;

	g_pDlg->m_setting.First_Contact.store(true);
	int curve_traj_iter = 0;

	// 경로 데이터 총 크기 계산
	int Total_size = g_pDlg->m_traj_data.size();

	float v_tcp[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f };

	// 데이터 기록 시작
	g_pDlg->m_flags.logThreadRunning.store(true);
	g_pDlg->m_pThread_Logger = AfxBeginThread(Thread_Logger, g_pDlg);

	t_start = system_clock::now();

	// 로봇 구동 루프 시작 (곡면 경로)
	while (g_pDlg->m_flags.robotRunning.load())
	{
		auto ts = std::chrono::steady_clock::now();

		// 로봇 & 센서 정보 수신
		auto Th_robotData_curve = g_pDlg->m_robotState.getSnapshot();
		auto Th_sensorData_curve = g_pDlg->m_ftSensor.getSnapshot();

		auto rot_matrix = Drfl.get_current_rotm();

		if (!g_pDlg->m_flags.curve_stop.load())
		{
			g_pDlg->m_servoctrl.vx_cmd = rot_matrix[0][0] * v_tcp[0] + rot_matrix[0][1] * v_tcp[1] + rot_matrix[0][2] * v_tcp[2];
			g_pDlg->m_servoctrl.vy_cmd = rot_matrix[1][0] * v_tcp[0] + rot_matrix[1][1] * v_tcp[1] + rot_matrix[1][2] * v_tcp[2];
			g_pDlg->m_servoctrl.vz_cmd = rot_matrix[2][0] * v_tcp[0] + rot_matrix[2][1] * v_tcp[1] + rot_matrix[2][2] * v_tcp[2];

			g_pDlg->m_servoctrl.roll_cmd = rot_matrix[0][0] * v_tcp[3] + rot_matrix[0][1] * v_tcp[4] + rot_matrix[0][2] * v_tcp[5];
			g_pDlg->m_servoctrl.pitch_cmd = rot_matrix[1][0] * v_tcp[3] + rot_matrix[1][1] * v_tcp[4] + rot_matrix[1][2] * v_tcp[5];
			g_pDlg->m_servoctrl.yaw_cmd = rot_matrix[2][0] * v_tcp[3] + rot_matrix[2][1] * v_tcp[4] + rot_matrix[2][2] * v_tcp[5];

			Status_gui_str.Format(_T("곡면 경로 구동중..."));
			g_pDlg->var_status_gui.SetWindowTextW(Status_gui_str);
		}
		else
		{
			Status_gui_str.Format(_T("곡면 경로 구동 정지!!!"));
			g_pDlg->var_status_gui.SetWindowTextW(Status_gui_str);
			break;
		}

		//////////////////////////////////////////////////////////////////////////
		////데이터 기록+++++++++++++++++++++++++++++++++++++++++++++++++++++
		//////////////////////////////////////////////////////////////////////////		
		t_stamp_ns = system_clock::now() - t_start;
		t_stamp_ns_float = float(t_stamp_ns.count());
		float t_stamp_ms_float = t_stamp_ns_float / 1000000.0f;

		LogData log;
		log.data.reserve(41); // 로그 데이터 크기 예약

		log.data.push_back(t_stamp_ms_float);
		log.data.insert(log.data.end(), Th_robotData_curve.flangePos.begin(), Th_robotData_curve.flangePos.end());
		log.data.insert(log.data.end(), Th_robotData_curve.jointAng.begin(), Th_robotData_curve.jointAng.end());
		log.data.insert(log.data.end(), Th_robotData_curve.tcpVel.begin(), Th_robotData_curve.tcpVel.end());
		log.data.insert(log.data.end(), Th_robotData_curve.jointVel.begin(), Th_robotData_curve.jointVel.end());
		log.data.insert(log.data.end(), Th_sensorData_curve.filteredForce.begin(), Th_sensorData_curve.filteredForce.end());
		log.data.insert(log.data.end(), Th_sensorData_curve.filteredTorque.begin(), Th_sensorData_curve.filteredTorque.end());

		log.data.push_back(g_pDlg->m_setting.vz_mms);
		log.data.push_back((float)g_pDlg->m_pidctrl.getLastError());
		log.data.push_back((float)g_pDlg->m_airctrl.sendChamberPressure());
		log.data.push_back((float)g_pDlg->m_airctrl.sendChamberVoltage());
		log.data.push_back((float)g_pDlg->m_airctrl.sendSpindlePressure());
		log.data.push_back((float)g_pDlg->m_airctrl.sendSpindleVoltage());
		log.data.push_back((float)g_pDlg->m_airctrl.feedbackChamberPressure());
		log.data.push_back((float)g_pDlg->m_airctrl.feedbackChamberVoltage());
		log.data.push_back((float)g_pDlg->m_airctrl.feedbackSpindlePressure());
		log.data.push_back((float)g_pDlg->m_airctrl.feedbackSpindleVoltage());
		{
			std::lock_guard<std::mutex> lock(g_pDlg->m_logMutex);
			g_pDlg->m_logQueue.push(log);
		}
		g_pDlg->m_logCv.notify_one();

		////////////////////////////////////////////////////////////////////////////
		//////while문 속도 제어+++++++++++++++++++++++++++++++++++++++++++++++++++++
		////////////////////////////////////////////////////////////////////////////
		// 1ms 경과까지 스핀 대기
		for (;;)
		{
			LARGE_INTEGER now;
			QueryPerformanceCounter(&now);					// 현재 시각
			if (now.QuadPart >= nextTick.QuadPart) break;	// 1ms 경과
			_mm_pause();									// 20~30 ns 휴식
		}

		// 다음 루프를 위해 nextTick 업데이트
		nextTick.QuadPart += one_ms_tick;

		////////////////////////////////////////////////////////////////////////////
		//////주기 측정 및 출력+++++++++++++++++++++++++++++++++++++++++++++++++++++
		////////////////////////////////////////////////////////////////////////////
		i_freq++;
		if (i_freq == g_pDlg->m_setting.set_Hz)
		{
			LARGE_INTEGER now;
			QueryPerformanceCounter(&now);
			double elapsed_s = (double(now.QuadPart - freqStart.QuadPart) / double(qpf.QuadPart));
			g_pDlg->var_freq_main = 1.0 / (elapsed_s / static_cast<double>(g_pDlg->m_setting.set_Hz));
			i_freq = 0;
			freqStart = now;
		}
	}
	g_pDlg->m_flags.robotRunning.store(false);
	return 0;
}


UINT CRobotCommSWDJv5Dlg::Thread_Contact_Flat_RL(LPVOID pParam)
{
	CRobotCommSWDJv5Dlg* g_pDlg = (CRobotCommSWDJv5Dlg*)pParam;

	// 로봇 구동 쓰레드 우선순위 설정
	HANDLE hThread = GetCurrentThread();
	SetThreadPriority(hThread, THREAD_PRIORITY_HIGHEST);
	SetThreadAffinityMask(hThread, 1ULL << 2);					// CPU #2번(core 2) 고정

	// 데이터 저장용 시간 변수
	system_clock::time_point t_start, t_cd;
	nanoseconds t_stamp_ns, t_stamp_cd_ns;
	float t_stamp_ns_float = 0.0;
	float t_stamp_cd_ns_float = 0.0;

	// ===============================================
	// 고분해능 타이머 초기화
	// QueryPerformanceFrequency → 1초당 tick 수
	LARGE_INTEGER qpf;
	QueryPerformanceFrequency(&qpf);
	const LONGLONG one_ms_tick = qpf.QuadPart / 1000LL;

	// QueryPerformanceCounter → 시작 시각
	LARGE_INTEGER startTick;
	QueryPerformanceCounter(&startTick);		// 시작 시각
	// nextTick: 첫 목표 시각 = startTick + one_ms_tick
	LARGE_INTEGER nextTick = startTick;
	nextTick.QuadPart += one_ms_tick;		// 첫 주기 목표
	// 반복 주파수 측정용 변수(1초마다 실제 주기 계산)
	LARGE_INTEGER freqStart = startTick;
	int i_freq = 0;

	// ===============================================
	// 실험 설정
	g_pDlg->m_setting.Target_Force_N.store(-45.0f);				// 준영씨 수정 값(초기 목표 접촉력) 목표 접촉력 설정 (N)   [음수로 설정해줘야 로봇 베이스 좌표계와 동일한 방향]
	g_pDlg->m_setting.Force_limit_N.store(85.0f);				// 접촉력 제한값 설정 (N)
	g_pDlg->m_setting.Target_vz.store(5.0f);					// 목표 접촉 속도 설정 (mm/s) [양수로 설정해줘야 로봇 베이스 좌표계와 동일한 방향]
	int Saturation_time = 5000;									// 접촉 유지 시간 설정 (ms)
	g_pDlg->m_setting.First_Contact.store(true);

	// ==============================================
	// 로봇 동작 시 초기 로봇의 시작 및 종료에 대한 기준 위치값 측정
	auto Init_flange_pos_load = g_pDlg->m_robotState.getSnapshot();
	g_pDlg->m_robotState.saveInitialflangePosition(Init_flange_pos_load.flangePos.data(), 6);
	auto initialPosArray_flange = g_pDlg->m_robotState.getInitialflangePositionArray();

	// ===============================================
	// Cubic smoothstep 함수로 생성되는 속도 경로로 로봇을 하강 & 상승 구동하기 위한 설정
	static ProfileConfig approachConfig;
	static ProfileConfig retractConfig;
	static float start_pos_z = 0.0f;
	static auto rampStartTime = std::chrono::system_clock::now();

	// ===============================================
	// 공압 관련 변수 초기화
	const double pressureRampDownDuration_sec = 3.0;			// 압력 감소 시간 (초)
	static auto pressureRampStartTime = std::chrono::system_clock::now();
	float set_chamber_air = 0.02f;								// 준영씨 수정 값 (초기 공압 설정) 초기 챔버 공압 설정값 (MPa)
	
	g_pDlg->m_setting.save_init_air = set_chamber_air;

	// ===============================================
	// PID 게인 및 바운드 설정 
	// 준영씨 수정 값(초기 PID 게인값)
	g_pDlg->m_pidctrl.setGains(
		35.0,			// Kp
		35.0,			// Ki
		0.0);			// Kd
	g_pDlg->m_pidctrl.setOutputLimits(
		-1500.0,		// min
		1500.0);		// max

	double debug_p, debug_i, debug_d;

	// ===============================================
	// RL 통신용 변수 초기화
	float RL_precharge = 0.0f;									// 초기 공압값 초기화
	float RL_previous_error_force_z = 0.0f;						// 이전 접촉력 오차값 초기화
	float RL_integral_error_force_z = 0.0f;						// 접촉력 오차 적분값 초기화
	const float RL_dt = 0.001f;									// 제어 주기 (초 단위)
	bool RL_confirm = false;									// RL PC로부터 메세지 수신 확인 플래그 초기화
	bool episode_ended = false;									// 에피소드 종료 플래그 초기화
	bool RL_end = false;										// RL 학습 종료 플래그 초기화
	bool RL_prep_flag = false;									// PID 제어기 시작 3초전 플래그 초기화

	int RL_test_count = 1;										// C++ 제어기 내에서 현재 테스트 회차 카운트 변수

	// 적분 와인드업 방지를 위한 한계값
	const float RL_max_integral_limit = 100.0f;					// 최대 적분 한계값
	const float RL_min_integral_limit = -100.0f;				// 최소 적분 한계값

	int RL_count = 0;

	// 데이터 기록 시작
	g_pDlg->m_flags.logThreadRunning.store(true);
	g_pDlg->m_pThread_Logger = AfxBeginThread(Thread_Logger, g_pDlg);
	
	float Save_pos_z = 0.0f;
	float Save_Force_z = 0.0f;
	float Save_Target_Force_z = 0.0f;
	float Save_Current_air = 0.0f;

	// ===============================================
	// 메시지 수신 주기 측정을 위한 변수
	// ===============================================
	static bool is_first_message = true;
	static auto last_message_time = std::chrono::steady_clock::now();
	t_start = system_clock::now();

	printf("에피소드 학습을 위한 로봇 동작 시작!\n");

	while (g_pDlg->m_flags.robotRunning.load())
	{
		auto ts = std::chrono::steady_clock::now();

		// 로봇 & 센서 정보 수신
		auto Th_robotData_flat = g_pDlg->m_robotState.getSnapshot();
		auto Th_sensorData_flat = g_pDlg->m_ftSensor.getSnapshot();

		Save_pos_z = Th_robotData_flat.flangePos[2];
		Save_Force_z = Th_sensorData_flat.filteredForce[2];
		Save_Target_Force_z = g_pDlg->m_setting.Target_Force_N.load();

		// 현재 설정된 PID 게인값 확인
		g_pDlg->m_pidctrl.getGains(debug_p, debug_i, debug_d);
		
		// 읽어온 PID 게인값을 GUI 변수에 할당
		g_pDlg->var_gain_p_gui = debug_p;
		g_pDlg->var_gain_i_gui = debug_i;
		g_pDlg->var_gain_d_gui = debug_d;

		// =========================================================================
		// 1. 서버로부터 데이터 수신
		// =========================================================================
		// Episode가 종료된 경우 , RL Episode Flag가 true로 설정됨
		//	 => 목표 접촉력 변경
		if (g_pDlg->m_received_RL_timing_accurate.load() == true)
		{
			// 메세지 수신 주기 측정 로직 시작
			auto current_message_time = std::chrono::steady_clock::now();
			double reception_period_ms = 0.0;

			if (!is_first_message) {
				// 첫 메시지가 아니라면, 이전 메시지와의 시간 차이를 계산
				auto duration = std::chrono::duration<double, std::milli>(current_message_time - last_message_time);
				reception_period_ms = duration.count();
			}
			else {
				// 첫 메시지인 경우 플래그를 변경
				is_first_message = false;
			}
			// 다음 측정을 위해 현재 시각을 저장
			last_message_time = current_message_time;

			RL_precharge = g_pDlg->m_received_RL_precharge_pressure.load();
			RL_confirm = g_pDlg->m_received_RL_timing_accurate.load();
			episode_ended = g_pDlg->m_received_RL_episode_done.load();
			RL_end = g_pDlg->m_received_RL_learning_done.load();
			RL_count++;

			// 수신된 초기 공압값을 소수점 3자리로 반올림하여 설정
			// (네트워크 전송 시 float 정밀도 손실을 방지하고 명확한 제어값 사용)
			set_chamber_air = roundToDecimalPlaces(RL_precharge, 3);
		
			if (episode_ended)
			{ 
				g_pDlg->m_pidctrl.setGains(
					g_pDlg->m_received_RL_P_Gain.load(),
					g_pDlg->m_received_RL_I_Gain.load(),
					g_pDlg->m_received_RL_D_Gain.load()
				);
				RL_test_count++;
				printf("======================================\n");
				printf("현재 실험 회차: %d\n", RL_test_count);
				printf("수신된 Flag: %d\t%d\t%d\n", RL_confirm, episode_ended, RL_end);
				printf("수신된 공압 메세지: %.6f\t%.3f\t\n", RL_precharge, set_chamber_air);
				printf("수신된 PID 게인 메세지: %.3f\t%.3f\t%.3f\n",
					g_pDlg->m_received_RL_P_Gain.load(), g_pDlg->m_received_RL_I_Gain.load(), g_pDlg->m_received_RL_D_Gain.load());

				g_pDlg->m_setting.Target_Force_N.store(-45.0f);		// 준영씨 수정값 (에피소드 종료 후 초기 접촉값 설정)
				g_pDlg->m_airctrl.setDesiredChamberPressure(set_chamber_air);
				RL_count = 0;
				g_pDlg->m_received_RL_episode_done.store(false);

				g_pDlg->m_setting.First_Contact.store(true);
				g_pDlg->m_setting.Control_Step = 99;			// 에피소드 종료로 인한 환경 리셋
			}
			else if (RL_end == true)
			{
				g_pDlg->m_setting.First_Contact.store(true);
				g_pDlg->m_setting.Control_Step = 3;					// RL 학습 종료로 인한 로봇 구동 종료
			}
			else
			{
				if (reception_period_ms > 0.0) {
					printf("수신 주기: %.2f ms\n",
						RL_count, g_pDlg->m_received_RL_P_Gain.load(),g_pDlg->m_received_RL_I_Gain.load(),g_pDlg->m_received_RL_D_Gain.load(), reception_period_ms);
				}
				else {
					printf("RL PC로부터 %d번째 메세지 수신 성공! (RL_Gain: %.3f\t%.3f\t%.3f) | (첫 메시지)\n",
						RL_count, g_pDlg->m_received_RL_P_Gain.load(), g_pDlg->m_received_RL_I_Gain.load(), g_pDlg->m_received_RL_D_Gain.load());
				}
			}
			
			g_pDlg->m_received_RL_timing_accurate.store(false);
		}

		// =========================================================================
		// 2. 서버로 데이터 송신
		// =========================================================================
		// 서버로 보낼 메세지 생성
		float current_force_z = Th_sensorData_flat.filteredForce[2];
		float target_force_z = g_pDlg->m_setting.Target_Force_N.load();
		float error_force_z = target_force_z - current_force_z;
		float error_force_z_dot = (error_force_z - RL_previous_error_force_z) / RL_dt; // 접촉력 오차 미분값
		RL_previous_error_force_z = error_force_z;
		RL_integral_error_force_z += error_force_z * RL_dt;
		if (RL_integral_error_force_z > RL_max_integral_limit) {
			RL_integral_error_force_z = RL_max_integral_limit;
		}
		else if (RL_integral_error_force_z < RL_min_integral_limit) {
			RL_integral_error_force_z = RL_min_integral_limit;
		};
		auto current_chamber_p = (float)g_pDlg->m_airctrl.desiredChamberPressure();

		std::vector<char> packetToSend = PackRobotStatus(
			current_force_z,
			target_force_z,
			error_force_z,
			error_force_z_dot,
			RL_integral_error_force_z,
			current_chamber_p,
			g_pDlg->m_flags.RL_sanderactive_flag.load(),
			RL_precharge,
			Save_pos_z,
			RL_prep_flag);

		// 서버로 메세지 전송
		if (g_pDlg->m_tcpClient.IsConnected())
		{
			g_pDlg->m_tcpClient.Send(packetToSend.data(), packetToSend.size());
		}
		// =========================================================================
		// 3. 로봇 구동제어
		if (g_pDlg->m_flags.ftRunning.load())
		{
			if (g_pDlg->m_flags.flat_stop.load() == true)
			{
				g_pDlg->m_setting.vx_mms.store(0.0);
				g_pDlg->m_setting.vy_mms.store(0.0);
				g_pDlg->m_setting.vz_mms.store(0.0);

				g_pDlg->m_servoctrl.vx_cmd.store(g_pDlg->m_setting.vx_mms.load());
				g_pDlg->m_servoctrl.vy_cmd.store(g_pDlg->m_setting.vy_mms.load());
				g_pDlg->m_servoctrl.vy_cmd.store(g_pDlg->m_setting.vz_mms.load());

				g_pDlg->m_setting.Control_Step = 3;
				g_pDlg->m_setting.First_Contact.store(true);
				g_pDlg->m_flags.flat_stop.store(false);
			}

			// Control Step.0: 툴을 금형 시편에 접촉하기 위한 하강 동작
			if (g_pDlg->m_setting.Control_Step == 0)
			{
				if (g_pDlg->m_setting.First_Contact.load() == true) {
					// 1. 상태 초기화
					rampStartTime = std::chrono::system_clock::now();
					g_pDlg->m_setting.First_Contact.store(false);

					// 2. 하강 시작 위치 기록
					start_pos_z = Th_robotData_flat.flangePos[2];

					// 3. 하강 프로파일 설정
					approachConfig.direction = -1;
					approachConfig.max_velocity = g_pDlg->m_setting.Target_vz;
					approachConfig.final_velocity = 1.0f;
					approachConfig.target_z = 385.0f;
					approachConfig.ramp_duration_sec = 3.0; // 가속 시간 설정
					approachConfig.move_distance = std::abs(approachConfig.target_z - start_pos_z);	// 총 이동 거리 계산

					// 4. 챔버 공압 설정
					g_pDlg->m_airctrl.setDesiredChamberPressure(set_chamber_air);
				}
				auto now = std::chrono::system_clock::now();
				double elapsed_sec = std::chrono::duration<double>(now - rampStartTime).count();
				float current_pos_z = Th_robotData_flat.flangePos[2];

				// 설정된 approachConfig를 사용하여 속도 계산
				float final_target_vz = g_pDlg->m_velProfile.calculate(approachConfig, elapsed_sec, current_pos_z, start_pos_z);
				g_pDlg->m_setting.vz_mms.store(final_target_vz);

				// 목표 접촉력 도달 시 다음 단계로 이동
				if (Th_sensorData_flat.filteredForce[2] <= g_pDlg->m_setting.Target_Force_N)
				{
					g_pDlg->m_setting.Control_Step = 1;
					g_pDlg->m_setting.vz_mms.store(0.0f);
					g_pDlg->m_setting.First_Contact.store(true);
				}

				g_pDlg->m_servoctrl.vz_cmd.store(g_pDlg->m_setting.vz_mms);	// [mm/s]

				Status_gui_str.Format(_T("[평면 구동] Control Step 0:로봇 하강 중... (Vz: %.2f)"), -g_pDlg->m_setting.vz_mms.load());
				g_pDlg->var_status_gui.SetWindowTextW(Status_gui_str);
			}
			// Control Step.1: 일정 시간 동안 접촉 유지 (사용자가 정한 시간 동안)
			else if (g_pDlg->m_setting.Control_Step == 1)
			{
				if (g_pDlg->m_setting.First_Contact.load() == true)
				{
					t_cd = system_clock::now();
					g_pDlg->m_setting.First_Contact.store(false);
				}
				else
				{
					t_stamp_cd_ns = system_clock::now() - t_cd;
					t_stamp_cd_ns_float = float(t_stamp_cd_ns.count());
					float t_stamp_cd_ms_float = t_stamp_cd_ns_float / 1000000.0f;					

					// 목표 접촉 유지 시간 초과시 접촉 유지 정지 후 다음 단계로 이동
					if (t_stamp_cd_ms_float > Saturation_time)
					{
						g_pDlg->m_setting.vz_mms.store(0.0f);		// [mm/s] 접촉 유지 정지
						g_pDlg->m_setting.First_Contact.store(true);
						g_pDlg->m_setting.Control_Step = 2;						

						// 힘제어 시퀀스 시작 알림을 위한 서버로 메세지 전송
						g_pDlg->m_flags.RL_sanderactive_flag.store(true);
						RL_prep_flag = false;
					}
					// 초기 접촉력 수렴 2초 이후에 RL_prep_flag 활성화
					else if (t_stamp_cd_ms_float > 2000)
					{
						RL_prep_flag = true;
					}
					g_pDlg->m_setting.Contact_time = static_cast<int>(Saturation_time - t_stamp_cd_ms_float) * 0.001f;
				}

				float err_f = g_pDlg->m_setting.Target_Force_N.load() - Th_sensorData_flat.filteredForce[2];	// [N] 접촉력 오차
				float kp = 0.05f;

				g_pDlg->m_setting.vz_mms.store(kp * err_f);		// [mm/s] 접촉력 오차 기반 z방향 속도 설정
				g_pDlg->m_servoctrl.vz_cmd.store(g_pDlg->m_setting.vz_mms);	// [mm/s]

				Status_gui_str.Format(_T("[평면 구동] Control Step 1: 로봇 접촉 수렴중..."));
				g_pDlg->var_status_gui.SetWindowTextW(Status_gui_str);
			}
			// Control Step.2: PID 제어기를 적용한 공압 제어 및 평면 구동 시작
			else if (g_pDlg->m_setting.Control_Step == 2)
			{
				if (g_pDlg->m_setting.First_Contact.load() == true)
				{
					g_pDlg->m_setting.First_Contact.store(false);

					g_pDlg->m_airctrl.setDesiredSpindlePressure(0.0);

					// PID 컨트롤러 리셋
					g_pDlg->m_pidctrl.reset();
					g_pDlg->m_setting.Target_Force_N.store(-60.0f);		// 준영씨 수정 값 (바꿀 목표 접촉력)

					Status_gui_str.Format(_T("[평면 구동] Control Step 2: 평면 구동 & PID 힘 제어 시작"));
					g_pDlg->var_status_gui.SetWindowTextW(Status_gui_str);
				}
				else
				{
					static auto last_time = std::chrono::steady_clock::now();
					static bool is_first_run = true;

					auto current_time = std::chrono::steady_clock::now();
					double actual_dt = std::chrono::duration<double>(current_time - last_time).count();
					last_time = current_time;

					// PID 제어 주기 설정
					if (is_first_run)
					{
						actual_dt = 0.001;
						is_first_run = false;
					}
					else if (actual_dt > 0.01)
					{
						actual_dt = 0.01;
					}

					double current_force = Th_sensorData_flat.filteredForce[2];								// [N] 측정된 접촉력
					double setpoint_force = g_pDlg->m_setting.Target_Force_N.load();						// [N] 목표 접촉력

					// PID 제어기 계산
					PIDController::Result result = g_pDlg->m_pidctrl.calculate(setpoint_force, current_force, actual_dt);

					double force_correction_N = result.output;												// [N] PID 제어기 오차 기반 접촉력 보정값 (PID 출력값)

					// 접촉력 => 공압 변환
					double pressure_correction_mpa = -force_correction_N / g_pDlg->m_airctrl.getA_m2() * 1e-6;						// [MPa] 공압 보정값 (N -> MPa 변환)
					double new_target_pressure_mpa = g_pDlg->m_airctrl.get_base_pressure_mpa() + pressure_correction_mpa;			// [MPa] 새로운 목표 공압값 (기본 압력 + 보정값)
					new_target_pressure_mpa = std::clamp(new_target_pressure_mpa, 0.0, 0.4);				// [MPa] 공압 제한 (0.0 ~ 0.4 MPa)

					g_pDlg->m_airctrl.setDesiredChamberPressure(new_target_pressure_mpa);					// [MPa] 출력 챔버 공압 설정

					// 현재 로봇위치가 평면 구동 종료 위치를 넘어가는지 확인
					//   - 평면 구동 종료 위치는 사용자가 설정한 x_pos_bound[1] 값
					//   - 평면 구동 종료 위치에 도달하면 Control Step = 3 단계로 이동
					if (Th_robotData_flat.flangePos[0] >= g_pDlg->m_setting.x_pos_bound[1] + 70)
					{
						g_pDlg->m_setting.vx_mms.store(0.0);	// [mm/s]
						g_pDlg->m_setting.First_Contact.store(true);
						g_pDlg->m_setting.Control_Step = 3;
					}
				}

				g_pDlg->m_servoctrl.vx_cmd.store(g_pDlg->m_setting.vx_mms);									// [mm/s]
				g_pDlg->m_servoctrl.vz_cmd.store(g_pDlg->m_setting.vz_mms);									// [mm/s]

				// [MPa] 출력 챔버 공압 최종 설정 (PID 제어값 + RL 제어값)
				g_pDlg->m_airctrl.setDesiredChamberPressure(g_pDlg->m_airctrl.desiredChamberPressure());
			}
			// Control Step.3: 평면 경로 구동 마무리
			else if (g_pDlg->m_setting.Control_Step == 3)
			{
				if (g_pDlg->m_setting.First_Contact.load() == true)
				{
					// 1. 상태 초기화
					rampStartTime = system_clock::now();
					g_pDlg->m_setting.First_Contact.store(false);
					g_pDlg->m_flags.RL_sanderactive_flag.store(false);

					// 2. 동작 시작 시점의 상태 저장
					start_pos_z = Th_robotData_flat.flangePos[2];

					// 3. 상승(Retract) 프로파일 설정
					retractConfig.direction = 1;								// 상승
					retractConfig.max_velocity = g_pDlg->m_setting.Target_vz;	// 상승 속도
					retractConfig.final_velocity = 0.0f;						// 목표 지점에서 정지
					retractConfig.target_z = initialPosArray_flange[2];				// GUI에서 설정한 초기 Z 위치로 설정
					retractConfig.ramp_duration_sec = 3;						// 상승 가/감속 시간
					retractConfig.move_distance = std::abs(retractConfig.target_z - start_pos_z); // 총 이동 거리 계산

					// 4. 스핀들 공압 즉시 OFF
					g_pDlg->m_airctrl.setDesiredChamberPressure(0.0);			// 챔버 공압 OFF
					g_pDlg->m_airctrl.setDesiredSpindlePressure(0.0);			// 스핀들 공압 OFF

					Status_gui_str.Format(_T("[평면 구동] Control Step 3: 공압 제어 완료 및 초기 위치로 이동"));
					g_pDlg->var_status_gui.SetWindowTextW(Status_gui_str);
				}
				else
				{
					auto now = std::chrono::system_clock::now();
					float current_pos_z = Th_robotData_flat.flangePos[2];

					// 프로파일 기반 로봇 상승 로직
					double motion_elapsed_sec = std::chrono::duration<double>(now - rampStartTime).count();
					float final_target_vz = g_pDlg->m_velProfile.calculate(retractConfig, motion_elapsed_sec, current_pos_z, start_pos_z);
					g_pDlg->m_setting.vz_mms.store(final_target_vz);

					// 초기 로봇 시작 위치 이상으로 높은 위치에 TCP가 도달하면 구동 종료
					if (initialPosArray_flange[2] -5.0f <= Th_robotData_flat.flangePos[2])
					{
						g_pDlg->m_setting.vz_mms = 0.0;
						Status_gui_str.Format(_T("[평면 구동] Control Step 3: 로봇 구동 종료"));
						g_pDlg->var_status_gui.SetWindowTextW(Status_gui_str);

						g_pDlg->OnBnClickedButRobotDisconnect();
					}
				}
				g_pDlg->m_servoctrl.vz_cmd.store(g_pDlg->m_setting.vz_mms);	// [mm/s]
				Status_gui_str.Format(_T("[평면 구동] Control Step 3: 프로파일 상승 중... (Vz: %.2f)"), g_pDlg->m_setting.vz_mms.load());
				g_pDlg->var_status_gui.SetWindowTextW(Status_gui_str);
			}
			// Control Step.99: 에피소드 종료로 인한 환경 리셋
			else if (g_pDlg->m_setting.Control_Step == 99)
			{
				if (g_pDlg->m_setting.First_Contact.load() == true)
				{
					// 1. 상태 초기화
					rampStartTime = system_clock::now();
					g_pDlg->m_setting.First_Contact.store(false);
					g_pDlg->m_flags.RL_sanderactive_flag.store(false);
					
					// 2. 동작 시작 시점의 상태 저장
					start_pos_z = Th_robotData_flat.flangePos[2];

					// 3. 상승(Retract) 프로파일 설정
					retractConfig.direction = 1;								// 상승
					retractConfig.max_velocity = g_pDlg->m_setting.Target_vz;	// 상승 속도
					retractConfig.final_velocity = 0.0f;						// 목표 지점에서 정지
					retractConfig.target_z = 405.0f;							// 환경 리셋용 상승된 z 위치 설정
					retractConfig.ramp_duration_sec = 3;						// 상승 가/감속 시간
					retractConfig.move_distance = std::abs(retractConfig.target_z - start_pos_z); // 총 이동 거리 계산

					// 4. 스핀들 공압 즉시 OFF
					g_pDlg->m_airctrl.setDesiredChamberPressure(0.0);			// 챔버 공압 OFF
					g_pDlg->m_airctrl.setDesiredSpindlePressure(0.0);			// 스핀들 공압 OFF

					Status_gui_str.Format(_T("[평면 구동] Control Step 99: 에피소드 종료로 인한 환경 초기화 시작!"));
					g_pDlg->var_status_gui.SetWindowTextW(Status_gui_str);
				}
				else
				{
					auto now = std::chrono::system_clock::now();
					float current_pos_z = Th_robotData_flat.flangePos[2];

					// 프로파일 기반 로봇 상승 로직
					double motion_elapsed_sec = std::chrono::duration<double>(now - rampStartTime).count();
					float final_target_vz = g_pDlg->m_velProfile.calculate(retractConfig, motion_elapsed_sec, current_pos_z, start_pos_z);
					g_pDlg->m_setting.vz_mms.store(final_target_vz);

					// 초기 로봇 시작 위치 이상으로 높은 위치에 TCP가 도달하면 구동 종료
					if (395.0f  <= Th_robotData_flat.flangePos[2])
					{
						g_pDlg->m_setting.vz_mms = 0.0;
						g_pDlg->m_setting.Control_Step = 0;
						g_pDlg->m_setting.First_Contact.store(true);

						Status_gui_str.Format(_T("[평면 구동] Control Step 99: 환경 리셋 완료!!!"));
						g_pDlg->var_status_gui.SetWindowTextW(Status_gui_str);
					}
				}
				g_pDlg->m_servoctrl.vz_cmd.store(g_pDlg->m_setting.vz_mms);	// [mm/s]
				Status_gui_str.Format(_T("[평면 구동] Control Step 99: 환경 초기화를 위한 로봇 상승중... (Vz: %.2f)"), g_pDlg->m_setting.vz_mms.load());
				g_pDlg->var_status_gui.SetWindowTextW(Status_gui_str);
			}
		}

		//////////////////////////////////////////////////////////////////////////
		////데이터 기록+++++++++++++++++++++++++++++++++++++++++++++++++++++
		//////////////////////////////////////////////////////////////////////////		
		t_stamp_ns = system_clock::now() - t_start;
		t_stamp_ns_float = float(t_stamp_ns.count());
		float t_stamp_ms_float = t_stamp_ns_float / 1000000.0f;

		LogData log;
		log.data.reserve(5); // 로그 데이터 크기 예약

		log.data.push_back(t_stamp_ms_float);
		log.data.push_back(Save_pos_z);
		log.data.push_back(Save_Force_z);
		log.data.push_back(Save_Target_Force_z);
		log.data.push_back((float)g_pDlg->m_airctrl.feedbackChamberPressure());

		//log.data.insert(log.data.end(), Th_robotData_flat.flangePos.begin(), Th_robotData_flat.flangePos.end());
		//log.data.insert(log.data.end(), Th_sensorData_flat.filteredForce.begin(), Th_sensorData_flat.filteredForce.end());
		//log.data.insert(log.data.end(), Th_sensorData_flat.filteredTorque.begin(), Th_sensorData_flat.filteredTorque.end());

		//log.data.push_back(g_pDlg->m_setting.vz_mms);
		//log.data.push_back((float)g_pDlg->m_pidctrl.getLastError());
		//log.data.push_back((float)g_pDlg->m_airctrl.sendChamberPressure());
		//log.data.push_back((float)g_pDlg->m_airctrl.sendChamberVoltage());
		//log.data.push_back((float)g_pDlg->m_airctrl.sendSpindlePressure());
		//log.data.push_back((float)g_pDlg->m_airctrl.sendSpindleVoltage());
		//log.data.push_back((float)g_pDlg->m_airctrl.feedbackChamberPressure());
		//log.data.push_back((float)g_pDlg->m_airctrl.feedbackChamberVoltage());
		//log.data.push_back((float)g_pDlg->m_airctrl.feedbackSpindlePressure());
		//log.data.push_back((float)g_pDlg->m_airctrl.feedbackSpindleVoltage());

  //      log.data.push_back((float)g_pDlg->m_received_RL_Pressure);
		//log.data.push_back(RL_confirm);
		//log.data.push_back(episode_ended);
		{
			std::lock_guard<std::mutex> lock(g_pDlg->m_logMutex);
			g_pDlg->m_logQueue.push(log);
		}
		g_pDlg->m_logCv.notify_one();

		////////////////////////////////////////////////////////////////////////////
		//////while문 속도 제어+++++++++++++++++++++++++++++++++++++++++++++++++++++
		////////////////////////////////////////////////////////////////////////////
		// 1ms 경과까지 스핀 대기
		for (;;)
		{
			LARGE_INTEGER now;
			QueryPerformanceCounter(&now);					// 현재 시각
			if (now.QuadPart >= nextTick.QuadPart) break;	// 1ms 경과
			_mm_pause();									// 20~30 ns 휴식
		}

		// 다음 루프를 위해 nextTick 업데이트
		nextTick.QuadPart += one_ms_tick;

		////////////////////////////////////////////////////////////////////////////
		//////주기 측정 및 출력+++++++++++++++++++++++++++++++++++++++++++++++++++++
		////////////////////////////////////////////////////////////////////////////
		i_freq++;
		if (i_freq == g_pDlg->m_setting.set_Hz)
		{
			LARGE_INTEGER now;
			QueryPerformanceCounter(&now);
			double elapsed_s = (double(now.QuadPart - freqStart.QuadPart) / double(qpf.QuadPart));
			g_pDlg->var_freq_main = 1.0 / (elapsed_s / static_cast<double>(g_pDlg->m_setting.set_Hz));
			i_freq = 0;
			freqStart = now;
		}
	}
	g_pDlg->m_flags.logThreadRunning.store(false);
	g_pDlg->m_flags.robotRunning.store(false);
	return 0;
}

// =========================================
// 데이터 저장 쓰레드 함수
UINT CRobotCommSWDJv5Dlg::Thread_Logger(LPVOID pParam)
{
	CRobotCommSWDJv5Dlg* g_pDlg = (CRobotCommSWDJv5Dlg*)pParam;

	// 파일 오픈 및 파일 명 생성
	char filename[100];
	sprintf_s(filename, sizeof(filename), ".\\Data\\Data_Speedl_F_%.0f_P_%.2f_I_%.2f_D_%.2f_%.3f.csv", abs(g_pDlg->m_setting.Target_Force_N),
		g_pDlg->m_pidctrl.getKp(), g_pDlg->m_pidctrl.getKi(), g_pDlg->m_pidctrl.getKd(), g_pDlg->m_setting.save_init_air);

	g_pDlg->m_dataFile.open(filename);
	if (!g_pDlg->m_dataFile.is_open()) return 1;

	// 헤더 저장
	/*g_pDlg->m_dataFile << "time(ms),"
		<< "Pos1_act(mm),Pos2_act(mm),Pos3_act(mm),Pos4_act(deg),Pos5_act(deg),Pos6_act(deg),"
		<< "Force_x(N),Force_y(N),Force_z(N),Torque_x(Nmm),Torque_y(Nmm),Torque_z(Nmm),"
		<< "vz_mms(mm/s),PID_error_chamber(N),P_AO_chamber(MPa),V_AO_chamber(V),P_AO_spindle(MPa),V_AO_spindle(V),"
		<< "P_AI_chamber(MPa),V_AI_chamber(V),P_AI_spindle(MPa),V_AI_spindle(V),"
		<< "RL_pressure(MPa),RL_confirm_flag,RL_episode_flag\n";*/

	g_pDlg->m_dataFile << "time(ms),"
		<< "Pos_z(mm),Force_sensor(N),Target_Force(N),Current_Air(MPa)\n";

	while (g_pDlg->m_flags.logThreadRunning || !g_pDlg->m_logQueue.empty()) {
		std::unique_lock<std::mutex> lock(g_pDlg->m_logMutex);
		g_pDlg->m_logCv.wait(lock, [&] { return !g_pDlg->m_logQueue.empty() || !g_pDlg->m_flags.logThreadRunning; });

		// 큐에서 데이터 꺼내서 저장
		while (!g_pDlg->m_logQueue.empty()) {
			LogData log = g_pDlg->m_logQueue.front();
			g_pDlg->m_logQueue.pop();
			lock.unlock();


			for (size_t i = 0; i < log.data.size(); ++i)
			{
				g_pDlg->m_dataFile << log.data[i];
				// 마지막 데이터가 아니면 쉼표를 추가
				if (i < log.data.size() - 1)
				{
					g_pDlg->m_dataFile << ",";
				}
			}
			g_pDlg->m_dataFile << "\n";

			lock.lock();
		}
		g_pDlg->m_dataFile.flush();
	}

	g_pDlg->m_dataFile.close();
	return 0;
}

/////////////////////////////////////////////
// 5. UI 이벤트 핸들러 (버튼 이벤트 등)
/////////////////////////////////////////////

// =========================================
// 로봇 연결 버튼 클릭 이벤트 핸들러
void CRobotCommSWDJv5Dlg::OnBnClickedButRobotConnect()	
{
	// 연결 수립
	Status_gui_str.Format(_T("연결 시도중..."));
	var_status_gui.SetWindowTextW(Status_gui_str);

	bool bTcpOK = Drfl.open_connection("192.168.137.100");
	if (!bTcpOK) {
		Status_gui_str = _T("TCP/IP 연결 실패");
		var_status_gui.SetWindowTextW(Status_gui_str);
		return;
	}
	Status_gui_str = _T("TCP/IP 연결 성공");
	var_status_gui.SetWindowTextW(Status_gui_str);
	Sleep(1000);

	// 제어권 요청 및 서보 온
	Drfl.ManageAccessControl(MANAGE_ACCESS_CONTROL_FORCE_REQUEST);
	Drfl.set_digital_output(GPIO_CTRLBOX_DIGITAL_INDEX_1, 0);
	Status_gui_str = _T("제어권 이양");
	var_status_gui.SetWindowTextW(Status_gui_str);
	Sleep(1000);

	Drfl.set_robot_control(CONTROL_SERVO_ON);
	Status_gui_str = _T("Servo On");
	var_status_gui.SetWindowTextW(Status_gui_str);
	Sleep(1000);

	// 로봇 모드 확인 후 자동모드 전환
	if (Drfl.get_robot_state() == STATE_STANDBY) {
		Status_gui_str = _T("STANDBY 상태");
	}
	else {
		Status_gui_str = _T("기타 상태");
	}
	var_status_gui.SetWindowTextW(Status_gui_str);

	if (Drfl.get_robot_mode() == ROBOT_MODE_MANUAL) {
		Drfl.set_robot_mode(ROBOT_MODE_AUTONOMOUS);
		Status_gui_str = _T("자동모드로 전환");
		var_status_gui.SetWindowTextW(Status_gui_str);
		Sleep(1000);
	}

	// 2) UDP 리얼타임 채널 연결 (read_data_rt 사용)
	Status_gui_str = _T("Realtime Control(UDP) 연결 시도");
	var_status_gui.SetWindowTextW(Status_gui_str);

	bool bUdpOK = Drfl.connect_rt_control();							// UDP 채널 열기
	if (!bUdpOK) {
		Status_gui_str = _T("Realtime Control 연결 실패");
		var_status_gui.SetWindowTextW(Status_gui_str);
	}
	else {
		Drfl.set_rt_control_output("v1.0", 0.001f, /*losscount=*/4);	// 출력 데이터(로봇→PC) 송신 설정 (1 ms 주기)
		// 실시간 제어 시작
		Drfl.start_rt_control();

		m_flags.realtimeConnected.store(true);
		Status_gui_str = _T("Realtime Control 제어 시작");
		var_status_gui.SetWindowTextW(Status_gui_str);
	}

	SetTimer(1, 62, NULL);
}

// =========================================
// 로봇 연결 종료 버튼 클릭 이벤트 핸들러
void CRobotCommSWDJv5Dlg::OnBnClickedButRobotDisconnect()	
{
	// 1) 모든 쓰레드 안전 종료
	QuitAllThreads();

	// 2) TCP/IP 연결 종료
	m_tcpClient.Disconnect();

	// 3) DAQ Task 리소스 해제
	if (m_airctrl.isInitialized())
	{
		m_airctrl.releaseTasks();
	}

	// 4) FT 센서 연결 종료
	if (m_ftSensor.isConnected())
	{
		m_ftSensor.Release();
		printf("FT 센서 자원 해제 완료\n");
	}

	// 5) RT 제어 중단 -> 안전정지(Event Stop) -> Servo OFF
	Drfl.stop_rt_control();
	Drfl.disconnect_rt_control();
	Drfl.set_safety_mode(SAFETY_MODE_AUTONOMOUS,
		SAFETY_MODE_EVENT_STOP);
	Sleep(1000);
	Drfl.set_robot_control(CONTROL_RESET_SAFET_OFF);// Servo OFF	

	// 6) 타이머·이벤트 등 기타 자원 해제
	KillTimer(1);

	// 7) TCP/IP 연결 완전 종료
	Drfl.CloseConnection();

	Status_gui_str = _T("로봇-PC 연결을 안전하게 종료했습니다.");
	var_status_gui.SetWindowTextW(Status_gui_str);
}

// =========================================
// 로봇 위치 초기화 버튼 클릭 이벤트 핸들러
void CRobotCommSWDJv5Dlg::OnBnClickedButHomeInit()
{
	//float pos_reset[6] = { 90.0, 0.0, 85.0, -5.0, 90.0, 90.0 };
	float pos_reset[6] = { 90.0, 0.0, 90.0, 0.0, 90.0, -90.0 };
	int move_pos_reset = Drfl.movej(pos_reset, 5, 5);
	if (move_pos_reset)
	{
		Status_gui_str.Format(_T("엔드이펙터 초기 위치 리셋 완료"));
		var_status_gui.SetWindowTextW(Status_gui_str);
	}
	else
	{
		Status_gui_str.Format(_T("엔드이펙터 초기 위치 리셋 실패!!!"));
		var_status_gui.SetWindowTextW(Status_gui_str);
	}
}

// =========================================
// 로봇 홈 위치 이동 버튼 클릭 이벤트 핸들러	
void CRobotCommSWDJv5Dlg::OnBnClickedButHomeMove()
{
	// 초기화 자세 이동
	float home[6] = { 90.0, 0.0, 90.0, 0.0, 90.0, -90.0 };
	int move_home_msg1 = Drfl.movej(home, 15.0, 15.0);
	if (move_home_msg1)
	{
		Status_gui_str.Format(_T("초기화 자세 이동 완료"));
		var_status_gui.SetWindowTextW(Status_gui_str);
	}
	else
	{
		Status_gui_str.Format(_T("초기화 자세 이동 실패!!!"));
		var_status_gui.SetWindowTextW(Status_gui_str);
		return;
	}

	// 로봇 상태 정보 업데이트
	auto pRt = Drfl.read_data_rt();
	if (pRt) {
		float desVel[3] = { m_servoctrl.vx_cmd, m_servoctrl.vy_cmd, m_servoctrl.vz_cmd };
		m_robotState.updateAllRawData(
			pRt->actual_tcp_position,
			pRt->actual_flange_position,
			pRt->actual_joint_position,
			pRt->actual_tcp_velocity,
			pRt->actual_flange_velocity,
			pRt->actual_joint_velocity,
			desVel,
			6
		);
		m_robotState.updateSnapshot();
	}

	// 평면 구동 초기 위치 이동
	if (m_flags.loadTraj.load() == false)
	{
		auto robot_Data_home = m_robotState.getSnapshot();

		// 평면 구동 시작 위치 설정 & 이동
		float move_home2_pos[6] = {
			m_setting.x_pos_bound[0], robot_Data_home.flangePos[1], robot_Data_home.flangePos[2],
			robot_Data_home.flangePos[3], robot_Data_home.flangePos[4], robot_Data_home.flangePos[5] };
		float move_home2_vel[6] = { 200.0f, 200.0f, 200.0f, 200.0f, 200.0f, 200.0f };
		float move_home2_acc[6] = { 400.0f, 400.0f, 400.0f, 400.0f, 400.0f, 400.0f };

		int move_home_msg2 = Drfl.servol(move_home2_pos, move_home2_vel, move_home2_acc, 5);
		if (move_home_msg2)
		{
			Status_gui_str.Format(_T("평면 구동 위치 이동 완료"));
			var_status_gui.SetWindowTextW(Status_gui_str);	
		}
		else
		{
			Status_gui_str.Format(_T("평면 구동 위치 이동 실패!!!"));
			var_status_gui.SetWindowTextW(Status_gui_str);
		}
	}
	// =============================================
	// 곡면 구동 초기 위치 이동
	else
	{
		float move_home2_pos[6];
		memcpy(move_home2_pos, m_traj_data[0].joint_ang, sizeof(float) * 6);

		auto vel = m_servoctrl.home_servoj_vel;
		auto acc = m_servoctrl.home_servoj_acc;
		int move_home_msg2 = Drfl.movej(move_home2_pos, vel.data(), acc.data(), 10);
		if (move_home_msg2)
		{
			Status_gui_str.Format(_T("곡면 구동 위치 이동 완료"));
			var_status_gui.SetWindowTextW(Status_gui_str);
		}
		else
		{
			Status_gui_str.Format(_T("곡면 구동 위치 이동 실패!!!"));
			var_status_gui.SetWindowTextW(Status_gui_str);
		}		
	}

	// 최종 상태 업데이트
	pRt = Drfl.read_data_rt();
	if (pRt) {
		float desVel[3] = { m_servoctrl.vx_cmd, m_servoctrl.vy_cmd, m_servoctrl.vz_cmd };
		m_robotState.updateAllRawData(
			pRt->actual_tcp_position,
			pRt->actual_flange_position,
			pRt->actual_joint_position,
			pRt->actual_tcp_velocity,
			pRt->actual_flange_velocity,
			pRt->actual_joint_velocity,
			desVel,
			6
		);
		m_robotState.updateSnapshot();
	}
}

// =========================================
// 서보 제어 모드 시작 버튼 클릭 이벤트 핸들러
void CRobotCommSWDJv5Dlg::OnBnClickedButServo()
{
	// Servo 쓰레드 시작
	if (StartServoThread()) {
		Status_gui_str = _T("서보제어모드 시작");
		var_status_gui.SetWindowTextW(Status_gui_str);
		return;
	}
}

// =========================================
// 서보 제어 모드 정지 버튼 클릭 이벤트 핸들러
void CRobotCommSWDJv5Dlg::OnBnClickedButForceControl()
{
	// 서보 쓰레드가 실행 중이지 않다면 경고 알림 및 정지
	if (!m_flags.servoRunning.load()) {
		Status_gui_str = _T("서보제어모드를 먼저 시작하십시오");
		var_status_gui.SetWindowTextW(Status_gui_str);
		return;
	}
	// 공압 연결이 되어있지 않다면 자동으로 연결 (현재 공압 사용은 평면 구동 경로에만 사용)
	if (m_flags.loadTraj.load() == false)
	{
		if (!m_flags.airRunning.load()) {
			OnBnClickedButAirOn();
			Sleep(1000);
		}
	}
	// FT 센서 연결이 되어있지 않다면 자동으로 연결
	if (!m_flags.ftRunning.load()) {
		OnBnClickedButFtSensorOn();
		Sleep(1000);
	}

	OnBnClickedButTcpip();

	//========================================
	// 평면 경로 구동 조건
	if (m_flags.loadTraj.load() == false)
	{
		// Contact 쓰레드 시작 (Flat 경로 구동)
		//if (StartContactThread_Flat()) 
		//{
		//	Status_gui_str = _T("평면 경로 구동 쓰레드 시작!");
		//	var_status_gui.SetWindowTextW(Status_gui_str);
		//	SetTimer(1, 62, NULL);								// GUI 타이머
		//}

		// ================== 테스트 =================
		if (StartContactThread_Flat_RL())
		{
			Status_gui_str = _T("RL 기반 평면 경로 구동 쓰레드 시작!");
			var_status_gui.SetWindowTextW(Status_gui_str);
			SetTimer(1, 62, NULL);								// GUI 타이머
		}
		// ===========================================
	}
	//========================================
	// 곡면 경로 구동 조건
	else
	{
		// Contact 쓰레드 시작 (Curve 경로 구동)
		if (StartContactThread_Curve()) 
		{
			Status_gui_str = _T("곡면 경로 구동 쓰레드 시작!");
			var_status_gui.SetWindowTextW(Status_gui_str);
			SetTimer(1, 62, NULL);								// GUI 타이머
		}
	}
}

// =========================================
// GUI 업데이트 및 타이머 이벤트 핸들러
void CRobotCommSWDJv5Dlg::OnTimer(UINT_PTR nIDEvent)	
{
	// 1) GUI 업데이트
	UpdateGUI();			

	// 2) 그래프 업데이트
	auto gui_sensor = m_ftSensor.getSnapshot();
	auto gui_robot = m_robotState.getSnapshot();
	float forcePoints[2] = { fabs(m_setting.Target_Force_N), fabs(gui_sensor.filteredForce[2]) };
	float posPoints[2] = { m_servoctrl.servo_tcp_pos_des_key[2], gui_robot.tcpPos[2] };
	_rtGraphforce->AppendPoints(forcePoints);
	_rtGraphpos->AppendPoints(posPoints);

	CDialogEx::OnTimer(nIDEvent);
}

void CRobotCommSWDJv5Dlg::UpdateGUI()
{
	// 1) 주기 측정
	static int i_count = 0;
	static clock_t t_prev = clock();
	if (++i_count >= 10) {
		clock_t t_now = clock();
		double dt = double(t_now - t_prev) / CLOCKS_PER_SEC;
		if (dt > 0)
			var_freq_gui = round((10.0 / dt) * 100) / 100;
		t_prev = t_now;
		i_count = 0;
	}

	// 2) 데이터 스냅샷
	auto gui_robot = m_robotState.getSnapshot();
	auto gui_sensor = m_ftSensor.getSnapshot();

	// 3) 위치, 각도, 힘/토크, 속도 등을 포맷하여 출력
	for (int i = 0; i < 6; ++i) {
		// TCP 위치
		CString txt;
		txt.Format(_T("%.4f"), gui_robot.tcpPos[i]);
		GetDlgItem(IDC_VAR_POS1 + i)->SetWindowTextW(txt);
		// Joint 각도
		txt.Format(_T("%.4f"), gui_robot.jointAng[i]);
		GetDlgItem(IDC_VAR_ANG1 + i)->SetWindowTextW(txt);
		// FT 센서
		float ft = (i < 3) ? gui_sensor.filteredForce[i] : gui_sensor.filteredTorque[i - 3];
		txt.Format(_T("%.4f"), ft);
		GetDlgItem(IDC_VAR_FT1 + i)->SetWindowTextW(txt);
	}

	// 4) 속도 출력
	struct { UINT id; float v; } velItems[] = {
		{ IDC_VAR_INIT_X_VELOCITY_CURRENT, gui_robot.desVel[0] },
		{ IDC_VAR_INIT_Y_VELOCITY_CURRENT, gui_robot.desVel[1] },
		{ IDC_VAR_INIT_Z_VELOCITY_CURRENT, gui_robot.desVel[2] },
	};
	for (auto& it : velItems) {
		CString txt;
		txt.Format(_T("%.3f"), it.v);
		GetDlgItem(it.id)->SetWindowTextW(txt);
	}

	// 5) 기타 값 출력 (목표 힘, 접촉 시간, 공압 등)
	struct { UINT id; float v; } miscItems[] = {
		{ IDC_VAR_TARGET_FORCE,    fabs(m_setting.Target_Force_N) },
		{ IDC_VAR_CONTACT_TIMER,   m_setting.Contact_time },
		{ IDC_VAR_CHAMBER_AIR,     (float)m_airctrl.feedbackChamberPressure() },
		{ IDC_VAR_CHAMBER_VOLT,    (float)m_airctrl.feedbackChamberVoltage() },
		{ IDC_VAR_SPINDLE_AIR,     (float)m_airctrl.feedbackSpindlePressure() },
		{ IDC_VAR_SPINDLE_VOLT,    (float)m_airctrl.feedbackSpindleVoltage() },
		{ IDC_VAR_FREQ_GUI,        (float)var_freq_gui },
		{ IDC_VAR_FREQ_SERVO,      (float)var_freq_servo },
		{ IDC_VAR_FREQ_FT_SENSOR,  (float)var_freq_ft_sensor },
		{ IDC_VAR_FREQ_MAIN,       (float)var_freq_main },
		{ IDC_VAR_GAIN_P,          (float)var_gain_p_gui },
		{ IDC_VAR_GAIN_I,          (float)var_gain_i_gui },
		{ IDC_VAR_GAIN_D,          (float)var_gain_d_gui}
	};
	for (auto& it : miscItems) {
		CString txt;
		txt.Format(_T("%.3f"), it.v);
		GetDlgItem(it.id)->SetWindowTextW(txt);
	}
}

// =========================================
// 대화상자 종료 이벤트 핸들러
void CRobotCommSWDJv5Dlg::OnDestroy()	
{
	CDialogEx::OnDestroy();

	OnBnClickedButRobotDisconnect();

	if (m_ftSensor.isConnected())
	{
		m_ftSensor.Release();
		printf("FT 센서 연결 종료 완료\n");
	}

	delete _rtGraphforce;
	delete _rtGraphpos;
}

// =========================================
// 대화상자 닫기 이벤트 핸들러
void CRobotCommSWDJv5Dlg::OnClose()
{
	// 모든 통신 연결 종료
	OnBnClickedButRobotDisconnect();
	CDialogEx::OnClose();
}


/////////////////////////////////////////////
// 6. FT 센서 관련 함수
/////////////////////////////////////////////

// =========================================
// FT 센서 쓰레드 시작 버튼 클릭 이벤트 핸들러
void CRobotCommSWDJv5Dlg::OnBnClickedButFtSensorOn()	
{
	// 센서 쓰레드가 이미 돌고 있으면 재시작하지 않음
	if (m_pThread_FT) {
		Status_gui_str = _T("FT 센서가 이미 실행 중입니다.");
		var_status_gui.SetWindowTextW(Status_gui_str);
		return;
	}

	// 어댑터·CAN 초기화
	m_ftSensor.initializeSensor();

	BOOL fUserSelect = TRUE;
	auto& sensor = m_ftSensor;

	// 센서 선택 및 초기화
	HRESULT hr = sensor.SelectDevice(fUserSelect);
	if (hr == VCI_OK)
	{
		Status_gui_str = _T("FT 어댑터 선택 OK");
		var_status_gui.SetWindowTextW(Status_gui_str);

		hr = sensor.CheckBalFeatures(0);
		if (hr == VCI_OK)
		{
			Status_gui_str = _T("기능 확인 OK");
			var_status_gui.SetWindowTextW(Status_gui_str);
		}
		else
		{
			Status_gui_str.Format(_T("기능 확인 실패: 0x%08lX"), hr);
			var_status_gui.SetWindowTextW(Status_gui_str);
		}

		Status_gui_str = _T("CAN 초기화 중...");
		var_status_gui.SetWindowTextW(Status_gui_str);
		hr = sensor.InitSocket(0);

		if (hr == VCI_OK)
		{
			Status_gui_str = _T("CAN 초기화 OK");
			var_status_gui.SetWindowTextW(Status_gui_str);

			Status_gui_str = _T("바이어스 평균값 계산 중...");
			var_status_gui.SetWindowTextW(Status_gui_str);

			const int Bias_Cnt = 1000;										// 바이어스 평균값 계산 횟수
			std::array<float, 3> F_Bias_sum = { 0.0f, 0.0f, 0.0f };
			std::array<float, 3> T_Bias_sum = { 0.0f, 0.0f, 0.0f };

			// 바이어스 평균값 계산을 위한 센서 데이터 수집
			for (int i = 0; i < Bias_Cnt; ++i)
			{
				m_ftSensor.ProcessMessage(100);								// 센서 데이터 갱신
				auto rawForce = m_ftSensor.getState().getRawForce();
				auto rawTorque = m_ftSensor.getState().getRawTorque();

				for (int j = 0; j < 3; ++j)
				{
					F_Bias_sum[j] += rawForce[j];
					T_Bias_sum[j] += rawTorque[j];
				}
				Sleep(1);
			}

			std::array<float, 3> F_Bias, T_Bias;
			for (int j = 0; j < 3; ++j)
			{
				F_Bias[j] = F_Bias_sum[j] / Bias_Cnt;
				T_Bias[j] = T_Bias_sum[j] / Bias_Cnt;
			}

			// 바이어스 평균값 설정
			sensor.getState().setBiasForce(F_Bias);
			sensor.getState().setBiasTorque(T_Bias);

			Sleep(1000);

			// =========================================
			// FT 센서 쓰레드 시작
			if (StartFTThread())
			{
				SetTimer(1, 62, NULL);										// GUI 타이머 시작
				Status_gui_str = _T("FT센서 통신 연결 성공");
				var_status_gui.SetWindowTextW(Status_gui_str);
			}
			else
			{
				Status_gui_str = _T("FT센서 쓰레드 시작 실패!!!");
				var_status_gui.SetWindowTextW(Status_gui_str);
			}
		}
	}
	else
	{
		Status_gui_str = _T("FT센서 통신 연결 실패");
		var_status_gui.SetWindowTextW(Status_gui_str);
		sensor.Release();
		return;
	}
}

/////////////////////////////////////////////
// 7. 공압 제어 관련 함수
/////////////////////////////////////////////
// 
// =========================================
// 공압 제어 쓰레드 시작 버튼 클릭 이벤트 핸들러
void CRobotCommSWDJv5Dlg::OnBnClickedButAirOn()		
{
	m_flags.airTask.store(true);									// 공압 제어 태스크 활성화 플래그 설정	

	m_airctrl.setDesiredChamberPressure(0.022);						// 초기 압력 설정
	m_airctrl.setDesiredSpindlePressure(0.0);						// 초기 스핀들 압력 설정

	// =========================================
	// 공압 제어 쓰레드 시작
	
	if (StartAirThread()) 
	{
		Status_gui_str = _T("공압 제어 쓰레드 시작!");
	}
	 else if (m_pThread_Air)
	 {
		 Status_gui_str = _T("공압 쓰레드 시작 실패(이미 실행 중...)");
	 }
	var_status_gui.SetWindowTextW(Status_gui_str);
}

/////////////////////////////////////////////
// 8. 구동 종료 관련 함수
/////////////////////////////////////////////

// =========================================
// 로봇 정지 버튼 클릭 이벤트 핸들러
void CRobotCommSWDJv5Dlg::OnBnClickedButRobotStop()
{
	// 평면 구동 조건인 경우
	if (m_flags.loadTraj.load() == false)
	{
		// 공압 출력 종료 & 평면 구동 쓰레드 종료 신호 전달
		//   - 이미 태스크가 꺼져있다면 아무것도 하지 않고 함수를 종료 (오작동 방지)
		if (m_flags.airTask.load() == false)
			return;

		m_flags.airTask.store(false);										// 공기 제어 태스크 비활성화 플래그 설정
		m_flags.airRunning.store(false);									// 공압 쓰레드 정지 & 평면 구동 쓰레드 정지 플래그 설정

		// 1) 쓰레드 안전 종료
		SafeThreadQuit(m_pThread_Air, hThread_Air, m_flags.airRunning);

		// 2) 압력 0 MPa 출력 후 DAQ Task 해제
		if (m_airctrl.isInitialized()) {
			// 목표 압력을 0으로 설정  
			m_airctrl.setDesiredChamberPressure(0.0);
			m_airctrl.setDesiredSpindlePressure(0.0);

			// 설정된 0 압력을 실제 하드웨어로 전송  
			m_airctrl.updateOutputs();

			// DAQ 태스크(Chamber/Spindle) 모두 정리  
			m_airctrl.releaseTasks();
		}
		Status_gui_str = _T("공압 제어 태스크 안전 종료");
		var_status_gui.SetWindowTextW(Status_gui_str);
		
		m_flags.flat_stop.store(true);
	}
	// 곡면 구동 조건인 경우
	else if (m_flags.loadTraj.load() == true)
	{
		m_flags.curve_stop.store(true);												// 곡면 구동 쓰레드 정지 플래그 설정
		SafeThreadQuit(m_pThread_Servo, hThread_Servo, m_flags.servoRunning);		// 서보 쓰레드 종료
		Drfl.stop_rt_control();														// RT 제어 중단
	}
}

/////////////////////////////////////////////
// 9. 추가 기능 함수 구현
/////////////////////////////////////////////

// =========================================
// 평면 구동 경로 불러오기 버튼 클릭 이벤트 핸들러
void CRobotCommSWDJv5Dlg::OnBnClickedButLoadTrajFlat()
{
	m_flags.loadTraj.store(false);
}

// =========================================
// 곡면 구동 경로 불러오기 버튼 클릭 이벤트 핸들러
void CRobotCommSWDJv5Dlg::OnBnClickedButLoadTrajCurve()
{
	std::string traj_path = ".\\Trajectory\\traj_curve.txt";						// 곡면 시편 경로 파일명 및 경로 설정
	std::lock_guard<std::mutex> lock(m_trajMutex);
	if (LoadTrajectoryData(traj_path, m_traj_data))										// 곡면 시편 경로 불러오기
	{
		if (!m_traj_data.empty())
		{
			Status_gui_str.Format(_T("곡면 시편 경로 불러오기 성공"));
			var_status_gui.SetWindowTextW(Status_gui_str);

			// 곡면 경로 구동에 대한 관절 최대 속도 및 가속도 설정
			m_servoctrl.home_servoj_vel = { 145.0, 145.0, 145.0, 145.0, 145.0, 145.0 };
			m_servoctrl.home_servoj_acc = { 900.0, 900.0, 900.0, 900.0, 900.0, 900.0 };
			
			m_flags.loadTraj.store(true);
		}
		else
		{
			Status_gui_str.Format(_T("곡면 시편 경로 불러오기 실패!!!"));
			var_status_gui.SetWindowTextW(Status_gui_str);
			return;
		}
	}
	else
	{
		printf("LoadTrajectoryData 실패: %s\n", traj_path.c_str());
	}
}

// =========================================
// 로봇 구동 제어 에러 처리 함수
void CRobotCommSWDJv5Dlg::HandleServoError(int servo_msg, const CString& errorType)
{
	// RT/Servo OFF 처리
	Drfl.stop_rt_control();
	Drfl.disconnect_rt_control();
	Drfl.set_safety_mode(SAFETY_MODE_AUTONOMOUS, SAFETY_MODE_EVENT_STOP);
	Drfl.set_robot_control(CONTROL_RESET_SAFET_OFF);

	OnBnClickedButRobotStop();					// 로봇 정지 버튼 클릭 이벤트 핸들러 호출
	QuitAllThreads();							// 모든 쓰레드 안전 종료

	Status_gui_str.Format(_T("%s 구동 제어 에러(%d) 감지 → Servo Off"), (LPCTSTR)errorType, servo_msg);

	var_status_gui.SetWindowTextW(Status_gui_str);
}

void CRobotCommSWDJv5Dlg::OnTcpDataReceived(const CString& message)
{
	// 연결 종료 메시지를 받은 경우
	if (message == _T("CONNECTION_CLOSED"))
	{
		Status_gui_str = _T("서버와의 연결이 끊어졌습니다.");
		var_status_gui.SetWindowTextW(Status_gui_str);

		::PostMessage(m_hWnd, WM_USER_TCP_DISCONNECT, 0, 0);
		return;
	}

	Status_gui_str.Format(_T("TCP 수신: %s"), message.GetString());
	var_status_gui.SetWindowTextW(Status_gui_str);
}

LRESULT CRobotCommSWDJv5Dlg::OnTcpDisconnect(WPARAM wParam, LPARAM lParam)
{
	Status_gui_str = _T("서버와의 연결이 끊어졌습니다. 리소스를 정리합니다.");
	var_status_gui.SetWindowTextW(Status_gui_str);

	// 이 함수는 수신 쓰레드와 완전히 독립적으로 실행되므로 안전합니다.
	m_tcpClient.Disconnect();

	return 0;
}

void CRobotCommSWDJv5Dlg::OnBnClickedButTcpip()
{
	if (m_tcpClient.IsConnected())
	{
		Status_gui_str = _T("이미 서버에 연결되어 있습니다.");
		var_status_gui.SetWindowTextW(Status_gui_str);
		return;
	}

	const CString server_ip = _T("192.168.0.17");	// Python 서버 IP
	const UINT server_port = 8888;					// Python 서버 포트

    Status_gui_str.Format(_T("%s:%u 서버에 연결 시도 중..."), server_ip.GetString(), server_port);
	var_status_gui.SetWindowTextW(Status_gui_str);

	// TcpClient의 Connect 메서드 호출 하나로 연결과 수신 쓰레드 시작이 모두 처리됩니다.
	if (m_tcpClient.Connect(server_ip, server_port))
	{
		Status_gui_str = _T("서버 연결 성공! 데이터 수신 대기 중...");
		var_status_gui.SetWindowTextW(Status_gui_str);
		m_flags.tcpip_flag.store(true);
	}
	else
	{
		Status_gui_str = _T("서버 연결에 실패했습니다.");
		var_status_gui.SetWindowTextW(Status_gui_str);
	}
}

void CRobotCommSWDJv5Dlg::OnBnClickedButTcpsend()
{
	//// 1. 서버에 연결되어 있는지 먼저 확인합니다.
	//if (!m_tcpClient.IsConnected())
	//{
	//	Status_gui_str = _T("서버에 연결되어 있지 않습니다. 먼저 TCP/IP 연결을 시도하세요.");
	//	var_status_gui.SetWindowTextW(Status_gui_str);
	//	return;
	//}

	//// 2. 헬퍼 함수를 호출하여 테스트용 데이터로 전송 패킷을 생성합니다.
	//std::vector<char> packetToSend = PackRobotStatus(
	//	-12.34f, 56.78f, 90.12f, -9.87f, 40.4f, 12.03f, 1
	//);

	//// 3. 생성된 이진 패킷을 전송합니다.
	//if (m_tcpClient.Send(packetToSend.data(), packetToSend.size()))
	//{
	//	// CString::Format을 사용하여 문자열을 안전하게 포맷팅합니다.
	//	Status_gui_str.Format(_T("테스트 이진 패킷(%d bytes)을 성공적으로 전송했습니다."), packetToSend.size());
	//	var_status_gui.SetWindowTextW(Status_gui_str);
	//}
	//else
	//{
	//	Status_gui_str = _T("테스트 패킷 전송에 실패했습니다.");
	//	var_status_gui.SetWindowTextW(Status_gui_str);
	//}
}

void CRobotCommSWDJv5Dlg::OnRlDataReceived(const RLAgentPacket& packet)
{
	// 모든 처리가 끝난 깨끗한 구조체를 바로 사용
	// 원자적 멤버 변수에 값 저장 (스레드 안전)
	//m_received_RL_Pressure.store(packet.RL_ResidualP);
	m_received_RL_precharge_pressure.store(packet.RL_precharge);
	m_received_RL_P_Gain.store(packet.RL_gain_P);
	m_received_RL_I_Gain.store(packet.RL_gain_I);
	m_received_RL_D_Gain.store(packet.RL_gain_D);
	m_received_RL_timing_accurate.store(packet.RL_timing_accurate == 1);
	m_received_RL_episode_done.store(packet.RL_episode_done == 1);
	m_received_RL_learning_done.store(packet.RL_learning_done == 1);
}