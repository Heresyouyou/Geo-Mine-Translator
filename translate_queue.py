"""
翻译队列模块 — 严格 FIFO, 线程安全

用法:
    from translate_queue import TranslateQueue
    
    q = TranslateQueue(worker_count=1)  # worker_count=1 保证严格 FIFO
    q.start()
    
    job_id = q.submit(pdf_path='/path/to/file.pdf', sender='youyou', context={...})
    # ... 等处理完
    result = q.get_result(job_id)  # {'status','output_path','error',...}
    
    q.stop()
"""
import threading
import queue
import time
import uuid
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


@dataclass(order=True)
class Job:
    """队列任务 — 带优先级的严格 FIFO"""
    sort_index: int = field(init=False, repr=False)
    job_id: str
    pdf_path: str
    sender: str = ''           # 发送者标识, 用于原路返回
    context: Dict[str, Any] = field(default_factory=dict)  # 额外上下文 (会话 ID, 消息时间等)
    submitted_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        # 单调递增序号 = 严格 FIFO
        self.sort_index = Job._counter()
    
    @staticmethod
    def _counter():
        with Job._lock:
            Job._n += 1
            return Job._n

Job._n = 0
Job._lock = threading.Lock()


class TranslateQueue:
    """
    严格 FIFO 翻译队列
    
    - 单 worker (worker_count=1): 最严格 FIFO, 一个处理完再下一个
    - 多 worker: 并发但可能乱序 (不推荐, 除非不在意顺序)
    
    内部用 queue.PriorityQueue (堆排序), Job 自带递增序号保证先入队先出队
    """
    
    def __init__(self, worker_count: int = 1, pdf_translator=None):
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._results: Dict[str, Dict] = {}     # job_id → result
        self._results_lock = threading.Lock()
        self._workers: List[threading.Thread] = []
        self._running = False
        self._stop_event = threading.Event()
        self._worker_count = worker_count
        self._pdf_translator = pdf_translator   # 注入 PDF 翻译器
        self._completed = threading.Event()      # 队列为空且所有 worker 空闲时 set
    
    # ── 对外 API ────────────────────────────────────────
    
    def start(self):
        """启动 worker 线程"""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        for i in range(self._worker_count):
            t = threading.Thread(target=self._worker_loop, name=f'TranslateWorker-{i}', daemon=True)
            t.start()
            self._workers.append(t)
        logger.info(f'Queue started with {self._worker_count} worker(s)')
    
    def stop(self, wait_jobs=True, timeout: float = 300):
        """
        停止队列
        
        Args:
            wait_jobs: True = 等队列清空再退出
            timeout: 最多等多少秒
        """
        self._stop_event.set()
        if wait_jobs:
            deadline = time.time() + timeout
            while not self._queue.empty() and time.time() < deadline:
                time.sleep(0.5)
        self._running = False
        # 放哨兵让 worker 跳出
        for _ in self._workers:
            try:
                self._queue.put(None)  # type: ignore
            except Exception:
                pass
        logger.info('Queue stopped')
    
    def submit(self, pdf_path: str, sender: str = '', context: Optional[Dict] = None) -> str:
        """
        提交一个翻译任务
        
        Returns:
            job_id: 可用于 get_result 查询
        """
        job_id = f'job_{uuid.uuid4().hex[:12]}'
        job = Job(job_id=job_id, pdf_path=pdf_path, sender=sender, context=context or {})
        self._queue.put(job)
        logger.info(f'Submitted {job_id}: {Path(pdf_path).name} ← {sender} (seq={job.sort_index})')
        return job_id
    
    def get_result(self, job_id: str) -> Optional[Dict]:
        """获取某个 job 的结果 (可能还没完成)"""
        with self._results_lock:
            return self._results.get(job_id)
    
    def wait_result(self, job_id: str, timeout: float = 600, poll_interval: float = 0.5) -> Optional[Dict]:
        """阻塞等待某个 job 完成 (status 为 done 或 error)"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = self.get_result(job_id)
            if r and r.get('status') in ('done', 'error'):
                return r
            time.sleep(poll_interval)
        return self.get_result(job_id)
    
    def pending_count(self) -> int:
        """队列中待处理数量 (不含正在处理的)"""
        return self._queue.qsize()
    
    def is_idle(self) -> bool:
        """队列空且没有正在处理的 worker"""
        return self._queue.empty() and all(
            not t.is_alive() or t.name.endswith('-idle') for t in self._workers
        )
    
    # ── 内部 ──────────────────────────────────────────────
    
    def _worker_loop(self):
        worker_name = threading.current_thread().name
        logger.info(f'{worker_name} started')
        
        while not self._stop_event.is_set():
            try:
                # 带超时的 get, 让循环能及时响应 stop_event
                job = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            # 哨兵
            if job is None:
                break
            
            self._process_job(job)
        
        logger.info(f'{worker_name} stopped')
    
    def _process_job(self, job: Job):
        """处理单个 job — 核心翻译逻辑"""
        logger.info(f'Processing {job.job_id} (seq={job.sort_index}): {Path(job.pdf_path).name}')
        
        # 先存一个 pending 状态
        self._set_result(job.job_id, {
            'status': 'processing',
            'job_id': job.job_id,
            'pdf_path': job.pdf_path,
            'sender': job.sender,
            'started_at': time.time(),
        })
        
        t0 = time.time()
        
        try:
            if not Path(job.pdf_path).exists():
                raise FileNotFoundError(f'PDF not found: {job.pdf_path}')
            
            # 调用翻译器
            if self._pdf_translator is None:
                raise RuntimeError('PDF translator not configured')
            
            output = self._pdf_translator.translate_pdf(
                pdf_path=job.pdf_path,
                job_id=job.job_id,
            )
            
            elapsed = time.time() - t0
            logger.info(f'✅ {job.job_id} done in {elapsed:.1f}s → {output}')
            
            self._set_result(job.job_id, {
                'status': 'done',
                'job_id': job.job_id,
                'pdf_path': job.pdf_path,
                'sender': job.sender,
                'output_path': output,
                'elapsed': round(elapsed, 2),
                'finished_at': time.time(),
            })
            
        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f'❌ {job.job_id} failed after {elapsed:.1f}s: {e}')
            self._set_result(job.job_id, {
                'status': 'error',
                'job_id': job.job_id,
                'pdf_path': job.pdf_path,
                'sender': job.sender,
                'error': str(e),
                'elapsed': round(elapsed, 2),
                'finished_at': time.time(),
            })
    
    def _set_result(self, job_id: str, result: Dict):
        with self._results_lock:
            self._results[job_id] = result
    
    def __len__(self):
        return self.pending_count()


# ── 极简用法 demo ─────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    
    class MockTranslator:
        """测试用假翻译器"""
        def translate_pdf(self, pdf_path, job_id):
            time.sleep(1)  # 模拟耗时
            out = pdf_path.replace('.pdf', '_zh.pdf')
            Path(out).write_text('mock translated content', encoding='utf-8')
            return out
    
    q = TranslateQueue(worker_count=1, pdf_translator=MockTranslator())
    q.start()
    
    # 投递 3 个任务
    tmp_dir = Path(__file__).parent / 'test_jobs'
    tmp_dir.mkdir(exist_ok=True)
    for i in range(3):
        pdf = tmp_dir / f'test_{i}.pdf'
        pdf.write_bytes(b'%PDF-1.4 test')
        jid = q.submit(str(pdf), sender=f'user_{i}')
        print(f'Submitted {jid}')
    
    # 等所有完成
    for jid in list(q._results.keys()):
        r = q.wait_result(jid, timeout=30)
        print(f'Result {jid}: {r}')
    
    q.stop()
    print('All done')
