import errno
import fcntl
import os
import random
import select
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from typing import TYPE_CHECKING, ClassVar
from unittest import TestCase

import gpiod
from gpiod.line import Direction, Edge, Value

from . import gpiosim
from .helpers import is_free_threaded

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

import logging
import sys

if is_free_threaded():
    logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)

logger = logging.getLogger(__name__)


class ThreadedChip(TestCase):
    """Tests opening, closing, and querying `Chip`s pointing to the same path in multiple threads"""

    NUM_THREADS: ClassVar[int]
    ITERATIONS: ClassVar[int]
    TIMEOUT: ClassVar[int]

    @classmethod
    def setUpClass(cls) -> None:
        cls.NUM_THREADS = min(32, (os.cpu_count() or 1) + 4)
        cls.ITERATIONS = 200 if is_free_threaded() else 20
        cls.TIMEOUT = 2

    def setUp(self) -> None:
        self.sim = gpiosim.Chip(
            num_lines=4, label="foobar", line_names={0: "l0", 1: "l1", 2: "l2", 3: "l3"}
        )
        self.chip = gpiod.Chip(self.sim.dev_path)

    def tearDown(self) -> None:
        self.sim = None  # type: ignore[assignment]
        self.chip = None  # type: ignore[assignment]

    def test_chip_functions(self) -> None:
        barrier = threading.Barrier(self.NUM_THREADS)

        def worker() -> None:
            for iteration in range(self.ITERATIONS):
                offset = random.randint(0, 3)
                # resync the threads every few iterations
                # let broken threads raise an error
                if iteration % 20 == 0:
                    barrier.wait(timeout=self.TIMEOUT)

                # sprinkle some timing chaos to change scheduling
                time.sleep(random.uniform(0, 0.001))
                with gpiod.Chip(self.sim.dev_path) as chip:
                    info = chip.get_info()
                    assert (info.name, info.label, info.num_lines) == (
                        self.sim.name,
                        "foobar",
                        4,
                    )
                    line_info = chip.get_line_info(f"l{offset}")
                    assert (line_info.offset, line_info.name) == (offset, f"l{offset}")

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            futures = [executor.submit(worker) for _ in range(self.NUM_THREADS)]
            for future in as_completed(futures):
                future.result(timeout=self.TIMEOUT)


class ThreadedChipShared(TestCase):
    """Tests querying a single `Chip` shared across multiple threads after closing"""

    NUM_THREADS: ClassVar[int]
    ITERATIONS: ClassVar[int]
    TIMEOUT: ClassVar[int]

    @classmethod
    def setUpClass(cls) -> None:
        cls.NUM_THREADS = min(32, (os.cpu_count() or 1) + 4)
        cls.ITERATIONS = 200 if is_free_threaded() else 20
        cls.TIMEOUT = 2

    def setUp(self) -> None:
        self.sim = gpiosim.Chip(num_lines=4, label="foobar")
        self.chip = gpiod.Chip(self.sim.dev_path)

    def tearDown(self) -> None:
        self.sim = None  # type: ignore[assignment]
        self.chip = None  # type: ignore[assignment]

    def test_chip_functions(self) -> None:
        barrier = threading.Barrier(self.NUM_THREADS)
        chip = gpiod.Chip(self.sim.dev_path)
        lock: AbstractContextManager[None | bool]
        if os.getenv("TESTS_NO_LOCKING"):
            lock = nullcontext()
            print("Running tests without locking")
        else:
            lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            with lock:
                info = chip.get_info()
                chip.close()
            assert (info.name, info.label, info.num_lines) == (
                self.sim.name,
                "foobar",
                4,
            )

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            futures = [executor.submit(worker) for _ in range(self.NUM_THREADS)]
            error_count = 0
            for future in as_completed(futures):
                try:
                    future.result(timeout=self.TIMEOUT)
                except gpiod.ChipClosedError:
                    error_count += 1
            assert error_count == self.NUM_THREADS - 1

    def test_chip_functions2(self) -> None:
        barrier = threading.Barrier(self.NUM_THREADS)
        chip = self.chip

        def worker(tid: int) -> None:
            for iteration in range(self.ITERATIONS):
                offset = random.randint(0, 3)
                # resync the threads every few iterations
                # let broken threads raise an error
                if iteration % 20 == 0:
                    barrier.wait(timeout=self.TIMEOUT)

                # sprinkle some timing chaos to change scheduling
                time.sleep(random.uniform(0, 0.001))
                try:
                    info = chip.watch_line_info(offset)
                    assert info.offset == offset
                except OSError as e:
                    if e.errno == errno.EBUSY:
                        retry_count = 0
                        while retry_count < 2:
                            retry_count += 1
                            try:
                                if retry_count > 0:
                                    logger.debug(
                                        f"{tid} stuck unwatching {offset} {retry_count}"
                                    )
                                    time.sleep(0)
                                chip.unwatch_line_info(offset)
                                break
                            except OSError as e:
                                pass

                info = chip.get_line_info(offset)
                assert info.offset == offset

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            futures = [executor.submit(worker, _) for _ in range(self.NUM_THREADS)]
            for future in as_completed(futures, self.TIMEOUT):
                future.result(timeout=self.TIMEOUT)

    def test_chip_functions3(self) -> None:
        barrier = threading.Barrier(self.NUM_THREADS)
        chip = gpiod.Chip(self.sim.dev_path)
        lock = threading.Lock()
        watching: set[int] = set()

        def worker() -> None:
            for iteration in range(self.ITERATIONS):
                offset = random.randint(0, 3)
                # resync the threads every few iterations
                # let broken threads raise an error
                if iteration % 20 == 0:
                    barrier.wait(timeout=self.TIMEOUT)

                # sprinkle some timing chaos to change scheduling
                time.sleep(random.uniform(0, 0.001))
                with lock:
                    if offset in watching:
                        chip.unwatch_line_info(offset)
                        watching.remove(offset)
                        info = chip.get_line_info(offset)
                    else:
                        info = chip.watch_line_info(offset)
                        watching.add(offset)
                assert info.offset == offset

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            futures = [executor.submit(worker) for _ in range(self.NUM_THREADS)]
            for future in as_completed(futures):
                future.result(timeout=self.TIMEOUT)


class ThreadedChipSharedQuery(TestCase):
    """Tests querying a single `Chip`s shared across multiple thread"""

    NUM_THREADS: ClassVar[int]
    ITERATIONS: ClassVar[int]
    TIMEOUT: ClassVar[int]

    @classmethod
    def setUpClass(cls) -> None:
        cls.NUM_THREADS = min(32, (os.cpu_count() or 1) + 4)
        cls.ITERATIONS = 200 if is_free_threaded() else 20
        cls.TIMEOUT = 2

    def setUp(self) -> None:
        self.sim = gpiosim.Chip(
            num_lines=4, label="foobar", line_names={0: "l0", 1: "l1", 2: "l2", 3: "l3"}
        )
        self.chip = gpiod.Chip(self.sim.dev_path)

    def tearDown(self) -> None:
        self.chip.close()
        self.sim = None  # type: ignore[assignment]
        self.chip = None  # type: ignore[assignment]

    def test_chip_functions(self) -> None:
        barrier = threading.Barrier(self.NUM_THREADS)
        chip = gpiod.Chip(self.sim.dev_path)
        lock: AbstractContextManager[None | bool]
        if os.getenv("TESTS_NO_LOCKING"):
            lock = nullcontext()
            print("Running tests without locking")
        else:
            lock = threading.Lock()

        def worker() -> None:
            for iteration in range(self.ITERATIONS):
                offset = random.randint(0, 3)
                # resync the threads every few iterations
                # let broken threads raise an error
                if iteration % 20 == 0:
                    barrier.wait(timeout=self.TIMEOUT)

                # sprinkle some timing chaos to change scheduling
                time.sleep(random.uniform(0, 0.001))
                with lock:
                    info = chip.get_info()
                assert (info.name, info.label, info.num_lines) == (
                    self.sim.name,
                    "foobar",
                    4,
                )
                with lock:
                    line_info = chip.get_line_info(f"l{offset}")
                assert (line_info.offset, line_info.name) == (offset, f"l{offset}")

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            futures = [executor.submit(worker) for _ in range(self.NUM_THREADS)]
            for future in as_completed(futures):
                future.result(timeout=self.TIMEOUT)


class ThreadedLineRequest(TestCase):
    """Tests querying a single `Chip`s shared across multiple thread"""

    NUM_THREADS: ClassVar[int]
    ITERATIONS: ClassVar[int]
    TIMEOUT: ClassVar[int]

    @classmethod
    def setUpClass(cls) -> None:
        cls.NUM_THREADS = min(32, (os.cpu_count() or 1) + 4)
        cls.ITERATIONS = 200 if is_free_threaded() else 20
        cls.TIMEOUT = 2

    def setUp(self) -> None:
        self.sim = gpiosim.Chip(
            num_lines=4, label="foobar", line_names={0: "l0", 1: "l1", 2: "l2", 3: "l3"}
        )
        self.chip = gpiod.Chip(self.sim.dev_path)

    def tearDown(self) -> None:
        self.chip.close()
        self.sim = None  # type: ignore[assignment]
        self.chip = None  # type: ignore[assignment]

    def test_chip_functions(self) -> None:
        barrier = threading.Barrier(self.NUM_THREADS)

        num_lines = self.chip.get_info().num_lines
        request = self.chip.request_lines(
            config={range(num_lines): gpiod.LineSettings(direction=Direction.OUTPUT)}
        )

        def worker(tid: int) -> None:
            for iteration in range(self.ITERATIONS):
                # distribute threads across number of lines
                offset = tid % num_lines
                # resync the threads every few iterations
                # let broken threads raise an error
                if iteration % 20 == 0:
                    barrier.wait(timeout=self.TIMEOUT)

                # sprinkle some timing chaos to change scheduling
                time.sleep(random.uniform(0, 0.001))

                # for tids < num_threads, dedicate them to reconfiguring lines
                if tid < num_lines:
                    new_dir = (
                        Direction.INPUT if iteration % 2 == 0 else Direction.OUTPUT
                    )
                    request.reconfigure_lines(
                        config={offset: gpiod.LineSettings(direction=new_dir)}
                    )
                else:
                    try:
                        if request.get_value(offset) == Value.ACTIVE:
                            request.set_value(offset, Value.INACTIVE)
                        else:
                            request.set_value(offset, Value.ACTIVE)
                    # set_value may raise a permission error when the pin is INPUT
                    except OSError:
                        pass

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            futures = [executor.submit(worker, id) for id in range(self.NUM_THREADS)]
            for future in as_completed(futures):
                future.result(timeout=self.TIMEOUT)
        request.release()

    def test_concurrent_close(self) -> None:
        barrier = threading.Barrier(self.NUM_THREADS)
        lock: AbstractContextManager[None | bool]
        if os.getenv("TESTS_NO_LOCKING"):
            lock = nullcontext()
            print("Running tests without locking")
        else:
            lock = threading.Lock()

        num_lines = self.chip.get_info().num_lines
        request = self.chip.request_lines(
            config={
                range(num_lines): gpiod.LineSettings(
                    direction=Direction.OUTPUT, output_value=Value.INACTIVE
                )
            }
        )

        def worker() -> None:
            barrier.wait()
            with lock:
                info = request.get_values(range(num_lines))
                request.release()
            for line in info:
                assert line == Value.INACTIVE

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            futures = [executor.submit(worker) for _ in range(self.NUM_THREADS)]
            error_count = 0
            for future in as_completed(futures):
                try:
                    future.result(timeout=self.TIMEOUT)
                except gpiod.RequestReleasedError:
                    error_count += 1
            assert error_count == self.NUM_THREADS - 1

    def test_concurrent_edge_event_reading2(self) -> None:
        num_lines = self.chip.get_info().num_lines
        req = self.chip.request_lines(
            config={
                range(num_lines): gpiod.LineSettings(
                    direction=Direction.INPUT, edge_detection=Edge.BOTH
                )
            }
        )

        # If read_edge_events() is blocking, threads will hang forever waiting
        # for events that don't exist during shutdown.
        flags = fcntl.fcntl(req.fd, fcntl.F_GETFL)
        fcntl.fcntl(req.fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        worker_barrier = threading.Barrier(self.NUM_THREADS)
        feeder_barrier = threading.Barrier(2)
        done_fd = os.eventfd(0)

        total = 0
        counter_lock = threading.Lock()
        req_lock = threading.Lock()

        poll = select.epoll()
        poll.register(req.fd, select.EPOLLIN)
        poll.register(done_fd, select.EPOLLIN)

        def reader_worker(tid: int) -> None:
            nonlocal total
            should_exit = False
            local_count = 0
            worker_barrier.wait(self.TIMEOUT)

            while not should_exit:
                events = poll.poll(self.TIMEOUT)

                for fd, _ in events:
                    if fd == done_fd:
                        should_exit = True
                        continue

                    if fd == req.fd:
                        try:
                            with req_lock:
                                # O_NONBLOCK prevents hanging
                                evs = req.read_edge_events()
                            if evs:
                                local_count += len(evs)
                                # add some thread jitter
                                time.sleep(random.uniform(0.0001, 0.001))
                        except OSError as e:
                            if e.errno == errno.EAGAIN:
                                continue
                            raise

            logger.debug(f"{tid} {local_count=}")
            with counter_lock:
                total += local_count

        def feeder(tid: int) -> None:
            offsets = list(range(tid, num_lines, 2))
            worker_barrier.wait(self.TIMEOUT)

            for i in range(int(self.ITERATIONS / 2)):
                offset = offsets[i % len(offsets)]
                for pull in [gpiosim.Chip.Pull.UP, gpiosim.Chip.Pull.DOWN]:
                    self.sim.set_pull(offset, pull)

                    # forcefully deschedule the thread and do a busy wait before
                    # firing off the next edge
                    time.sleep(random.uniform(0, 0.0001))

            feeder_barrier.wait(self.TIMEOUT)
            # Thread 0 signals done when all pulses have fired
            if tid == 0:
                os.eventfd_write(done_fd, 1)

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as ex:
            futures = [ex.submit(feeder, i) for i in range(2)]
            futures += [ex.submit(reader_worker, i) for i in range(2, self.NUM_THREADS)]

            for f in as_completed(futures):
                f.result(timeout=self.TIMEOUT)

        for fd in [req.fd, done_fd]:
            poll.unregister(fd)
        poll.close()
        os.close(done_fd)
        req.release()

        # expect at least 80% of the events to have been captured (UP _and_ DOWN)
        self.assertGreater(total, self.ITERATIONS * 2 * 0.8)

    def test_concurrent_info_event_reading(self) -> None:
        num_lines = self.chip.get_info().num_lines
        for offset in range(num_lines):
            self.chip.watch_line_info(offset)
        # If read_edge_events() is blocking, threads will hang forever waiting
        # for events that don't exist during shutdown.
        flags = fcntl.fcntl(self.chip.fd, fcntl.F_GETFL)
        fcntl.fcntl(self.chip.fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        worker_barrier = threading.Barrier(self.NUM_THREADS)
        feeder_barrier = threading.Barrier(2)
        done_fd = os.eventfd(0)

        total = 0
        counter_lock = threading.Lock()

        poll = select.epoll()
        poll.register(self.chip.fd, select.EPOLLIN)
        poll.register(done_fd, select.EPOLLIN)

        def reader_worker(tid: int) -> None:
            worker_barrier.wait(self.TIMEOUT)
            should_exit = False
            local_count = 0
            nonlocal total
            while not should_exit:
                time.sleep(random.uniform(0.0001, 0.001))
                events = poll.poll(self.TIMEOUT)

                for fd, _ in events:
                    if fd == done_fd:
                        should_exit = True
                        continue
                    if fd == self.chip.fd:
                        # read_info_event() only reads ONE event at a time (unlike edge events).
                        # We must loop until EAGAIN to fully drain the kernel buffer.
                        try:
                            while True:
                                _event = self.chip.read_info_event()
                                local_count += 1

                                # Stagger point: let other readers help drain
                                time.sleep(random.uniform(0.0001, 0.001))
                        except OSError as e:
                            if e.errno == errno.EAGAIN:
                                continue
                            raise

            logger.debug(f"{tid} {local_count=}")
            with counter_lock:
                total += local_count

        def feeder(tid: int) -> None:
            offsets = list(range(tid, num_lines, 2))
            worker_barrier.wait(self.TIMEOUT)

            for i in range(int(self.ITERATIONS / 2)):
                offset = offsets[i % len(offsets)]

                time.sleep(random.uniform(0.0001, 0.001))
                req = self.chip.request_lines(
                    config={offset: gpiod.LineSettings(direction=Direction.INPUT)}
                )

                req.reconfigure_lines(
                    config={offset: gpiod.LineSettings(direction=Direction.OUTPUT)}
                )

                req.release()

            feeder_barrier.wait(self.TIMEOUT)
            # Thread 0 signals done when all pulses have fired
            if tid == 0:
                os.eventfd_write(done_fd, 1)

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as ex:
            futures = [ex.submit(feeder, i) for i in range(2)]
            futures += [ex.submit(reader_worker, i) for i in range(2, self.NUM_THREADS)]

            for f in as_completed(futures, self.TIMEOUT):
                f.result(timeout=self.TIMEOUT)

        for fd in [self.chip.fd, done_fd]:
            poll.unregister(fd)
        poll.close()
        os.close(done_fd)
        for offset in range(num_lines):
            self.chip.unwatch_line_info(offset)

        # expect at least 80% of the events to have been captured (REQUEST/RECONFIGURE/RELEASE)
        self.assertGreater(total, self.ITERATIONS * 3 * 0.8)
