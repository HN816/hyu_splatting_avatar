import time
from gaussian_renderer import network_gui


def run_testing_live(pipe, frameset_test, gs_model, pc_dir, verify=None):
    for i, frame in enumerate(frameset_test):
        # custom_cam 등은 network_gui 내부에서 받아오기 때문에 임시 생성 필요 없음
        # 대신 render_to_network()를 참고하여 직접 렌더링 후 이미지 받기
        
        # render_to_camera()는 내부적으로 camera 객체가 필요하므로,
        # 네트워크에서 카메라 정보 받는 구조를 그대로 흉내내거나
        # 아니면 network_gui.render_to_network() 호출을 고려
        
        # 가장 쉬운 방법은 network_gui.render_to_network()를 그대로 호출해서 렌더링 + 전송 수행하는 것
        # 단, 이 함수는 내부에서 TCP와 통신하므로, verify 문자열 전달
        network_gui.render_to_network(gs_model, pipe, verify if verify else "verify")

        print(f"[run_testing_live] frame {i} sent")
        time.sleep(0.033)  # 30fps 조절