import tkinter as tk
from tkinter import scrolledtext, font as tkfont, messagebox
import subprocess
import threading
import re
import queue
import sys
import os

# conftest.py가 있는 경로를 import 경로에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

try:
    import conftest
except ImportError:
    messagebox.showerror("Error", "conftest.py 파일을 찾을 수 없습니다. 같은 디렉토리에 있는지 확인하세요.")
    sys.exit(1)
except Exception as e:
    messagebox.showerror("Error", f"conftest.py 로드 중 오류 발생:\n{e}")
    sys.exit(1)

# 상수 정의
WINDOW_TITLE = "ExecuTorch Model Verifier GUI"
GEOMETRY = "1000x700"
LOG_FONT = ("Courier New", 9)
RESULT_FONT = ("Segoe UI", 10)
RESULT_LABEL_FONT = ("Segoe UI", 10, "bold")
STATUS_RUNNING = "실행 중..."
STATUS_WAITING = "대기 중"
STATUS_DONE = "완료"
STATUS_FAILED = "실패"

# GUI 클래스
class VerifierApp:
    def __init__(self, master):
        self.master = master
        self.master.title(WINDOW_TITLE)
        self.master.geometry(GEOMETRY)

        self.model_configs = conftest.MODEL_CONFIGS 
        self.buttons = {}
        self.queue = queue.Queue() 

        self._setup_ui()
        self.master.after(100, self._process_queue) 

    def _setup_ui(self):
        # 메인 프레임 (좌: 모델 목록, 우: 로그/결과)
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 왼쪽: 모델 목록 프레임
        model_list_frame = tk.Frame(main_frame, width=250, relief=tk.GROOVE, borderwidth=1)
        model_list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        model_list_frame.pack_propagate(False) # 프레임 크기 고정

        tk.Label(model_list_frame, text="Model List", font=("Segoe UI", 12, "bold")).pack(pady=10)

        # 모델 버튼 및 설명 동적 생성
        for name, config in self.model_configs.items():
            item_frame = tk.Frame(model_list_frame)
            item_frame.pack(fill=tk.X, padx=10, pady=3)

            button = tk.Button(item_frame, text=name, width=15,
                               command=lambda m=name: self._start_test_thread(m))
            button.pack(side=tk.LEFT)
            self.buttons[name] = button

            # 모델 타입 설명 레이블 추가
            desc = f"({config.get('type', 'N/A')})"
            desc_label = tk.Label(item_frame, text=desc, anchor=tk.W, fg="gray")
            desc_label.pack(side=tk.LEFT, padx=(5, 0))


        # 오른쪽: 로그 및 결과 프레임
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 상태 표시 레이블
        self.status_label = tk.Label(right_frame, text=STATUS_WAITING, font=("Segoe UI", 11, "bold"))
        self.status_label.pack(anchor=tk.W, pady=(0, 5))

        # 로그 영역
        log_label = tk.Label(right_frame, text="Logs:")
        log_label.pack(anchor=tk.W)
        self.log_area = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, font=LOG_FONT, height=15)
        self.log_area.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.log_area.configure(state='disabled')

        # 결과 영역
        result_label_title = tk.Label(right_frame, text="--- Final Results ---", font=("Segoe UI", 11, "bold"))
        result_label_title.pack(anchor=tk.W)

        # 결과를 표시할 프레임
        self.result_frame = tk.Frame(right_frame)
        self.result_frame.pack(fill=tk.X, pady=(5,0))
        # 결과 표시용 레이블 미리 생성 (내용은 나중에 업데이트)
        self.result_labels = {
            "model_name": self._create_result_label_pair("Model Name (모델 이름):"),
            "model_type": self._create_result_label_pair("Model Type (모델 타입):"),
            "mae": self._create_result_label_pair("Mean Absolute Diff (평균 절대 오차):"),
            "torch_avg_latency": self._create_result_label_pair("PyTorch Avg Latency (평균 지연시간):"),
            "torch_max_latency": self._create_result_label_pair("PyTorch Max Latency (최대 지연시간):"),
            "et_avg_latency": self._create_result_label_pair("ExecuTorch Avg Latency (평균 지연시간):"),
            "et_max_latency": self._create_result_label_pair("ExecuTorch Max Latency (최대 지연시간):"),
        }

    def _create_result_label_pair(self, label_text):
        """ 결과 표시용 레이블 쌍 (설명 + 값) 생성 및 배치 """
        row_index = len(self.result_frame.grid_slaves()) # 현재 행 인덱스
        label = tk.Label(self.result_frame, text=label_text, font=RESULT_LABEL_FONT, anchor=tk.W)
        label.grid(row=row_index, column=0, sticky=tk.W, padx=(0, 10))
        value_label = tk.Label(self.result_frame, text="-", font=RESULT_FONT, anchor=tk.W)
        value_label.grid(row=row_index, column=1, sticky=tk.W)
        return value_label # 값 레이블만 반환

    def _log(self, message):
        """ 로그 영역에 메시지 추가 (스레드 안전) """
        self.log_area.configure(state='normal')
        self.log_area.insert(tk.END, message)
        self.log_area.see(tk.END) # 자동 스크롤
        self.log_area.configure(state='disabled')

    def _set_status(self, status, color="black"):
        """ 상태 레이블 업데이트 """
        self.status_label.config(text=status, fg=color)

    def _set_results(self, results_dict):
        """ 결과 영역 업데이트 """
        if results_dict:
            # 각 결과 레이블의 텍스트 업데이트
            self.result_labels["model_name"].config(text=results_dict.get('model_name', 'N/A'))
            self.result_labels["model_type"].config(text=results_dict.get('model_type', 'N/A'))
            self.result_labels["mae"].config(text=results_dict.get('mae', 'N/A'))
            self.result_labels["torch_avg_latency"].config(text=f"{results_dict.get('torch_avg', 'N/A')} ms")
            self.result_labels["torch_max_latency"].config(text=f"{results_dict.get('torch_max', 'N/A')} ms")
            self.result_labels["et_avg_latency"].config(text=f"{results_dict.get('et_avg', 'N/A')} ms")
            self.result_labels["et_max_latency"].config(text=f"{results_dict.get('et_max', 'N/A')} ms")
        else:
            for label in self.result_labels.values():
                label.config(text="-")

    def _toggle_buttons(self, enabled):
        """ 모든 모델 버튼 활성화/비활성화 """
        state = tk.NORMAL if enabled else tk.DISABLED
        for button in self.buttons.values():
            button.config(state=state)

    def _parse_results(self, output):
        """ pytest 출력에서 최종 결과 파싱 (개선된 버전) """
        results = {}
        torch_latency_match = re.search(r"pytorch_latency:\s*avg\s*([\d.]+)\s*ms,\s*max\s*([\d.]+)\s*ms", output)
        et_latency_match = re.search(r"executorch_latency:\s*avg\s*([\d.]+)\s*ms,\s*max\s*([\d.]+)\s*ms", output)
        mae_match = re.search(r"mean_absolute_difference:\s*([\d.e-]+)", output)
        name_match = re.search(r"model_name:\s*(\S+)", output)
        type_match = re.search(r"model_type:\s*(\S+)", output)

        if name_match: results['model_name'] = name_match.group(1).strip()
        if type_match: results['model_type'] = type_match.group(1).strip()
        if mae_match: results['mae'] = mae_match.group(1).strip()
        if torch_latency_match:
            results['torch_avg'] = torch_latency_match.group(1).strip()
            results['torch_max'] = torch_latency_match.group(2).strip()
        if et_latency_match:
            results['et_avg'] = et_latency_match.group(1).strip()
            results['et_max'] = et_latency_match.group(2).strip()

        return results if 'mae' in results else None

    def _run_pytest_thread(self, model_name):
        """ pytest 실행을 위한 스레드 함수 """
        try:
            self.queue.put(("status", STATUS_RUNNING, "blue"))
            self.queue.put(("buttons", False)) 
            self.queue.put(("clear_log", None))
            self.queue.put(("clear_results", None))

            # 현재 파이썬 인터프리터를 사용하여 pytest 모듈 실행 (가상환경 호환)
            command = [sys.executable, "-m", "pytest", "test.py", "-v", "-s", "--model", model_name]

            # subprocess.Popen으로 실시간 출력 캡처
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace',
                                       bufsize=1, universal_newlines=True)

            full_output = ""
            for line in process.stdout:
                self.queue.put(("log", line))
                full_output += line
                if line.startswith("[Setup]"):
                    status_msg = line.strip().replace("[Setup] ", "")
                    self.queue.put(("status", f"실행 중... ({status_msg})", "blue"))


            process.wait() # 프로세스 종료 대기

            if process.returncode == 0: # pytest 성공 (PASSED)
                parsed_results = self._parse_results(full_output)
                self.queue.put(("results", parsed_results))
                self.queue.put(("status", STATUS_DONE, "green"))
            else: # pytest 실패 (FAILED or ERROR)
                # 실패 시에도 결과 파싱 시도 (AssertionError인 경우 결과가 있을 수 있음)
                parsed_results = self._parse_results(full_output)
                self.queue.put(("results", parsed_results)) # 파싱된 결과(있다면) 표시
                self.queue.put(("status", STATUS_FAILED, "red"))


        except FileNotFoundError:
             self.queue.put(("log", f"Error: Cannot run '{sys.executable} -m pytest'. Is pytest installed in this environment?\n"))
             self.queue.put(("status", STATUS_FAILED, "red"))
        except Exception as e:
            self.queue.put(("log", f"An unexpected error occurred: {e}\n"))
            self.queue.put(("status", STATUS_FAILED, "red"))
        finally:
            self.queue.put(("buttons", True)) # 버튼 다시 활성화

    def _start_test_thread(self, model_name):
        """ 테스트 스레드 시작 """
        thread = threading.Thread(target=self._run_pytest_thread, args=(model_name,), daemon=True)
        thread.start()

    def _process_queue(self):
        """ 주기적으로 큐를 확인하여 GUI 업데이트 """
        try:
            while True: # 큐에 있는 모든 메시지 처리
                msg_type, value, *args = self.queue.get_nowait()
                color = args[0] if args else "black"

                if msg_type == "log":
                    self._log(value)
                elif msg_type == "status":
                    self._set_status(value, color)
                elif msg_type == "results":
                    self._set_results(value)
                elif msg_type == "buttons":
                    self._toggle_buttons(value)
                elif msg_type == "clear_log":
                    self.log_area.configure(state='normal')
                    self.log_area.delete('1.0', tk.END)
                    self.log_area.configure(state='disabled')
                elif msg_type == "clear_results":
                     # 결과 레이블 초기화
                     self._set_results(None)

        except queue.Empty:
            pass 
        finally:
            self.master.after(100, self._process_queue)

# 애플리케이션 실행
if __name__ == "__main__":
    root = tk.Tk()
    app = VerifierApp(root)
    root.mainloop()

