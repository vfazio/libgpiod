import errno
import fcntl
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from select import EPOLLIN, epoll
from typing import TYPE_CHECKING, ClassVar
from unittest import TestCase

import gpiod
from gpiod.line import Direction, Edge, Value

from . import gpiosim
from .helpers import is_free_threaded

if TYPE_CHECKING:
    from contextlib import AbstractContextManager


if is_free_threaded():
    logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)

logger = logging.getLogger(__name__)


# Threading & the CPython bindings as they relate to the C extension
#
# Of the objects exposed by the bindings, the following are effectively "frozen":
#   * ChipInfo
#   * LineInfo
#   * InfoEvent
#   * EdgeEvent
#   * gpiod.line Enums
#
# The *Info and *Event objects are return values from the C extension, are not
# inputs, and cannot be mutated. There should be no thread-safety concerns for
# these objects
#
# The remaining objects are:
#   * Chip
#   * LineRequest
#   * LineSettings
#
# LineSettings are a pure Python class, are arguments to functions, and are not
# passed to the C extension directly. There should be no major concerns about
# thread-safety within the C extension.
#
# Chip and LineRequest objects are pure Python classes _but_ they wrap classes
# that are exposed by the C extension. As such, they are at risk for conflicts
# between threads.
#
# There are levels of thread safety considerations here.
#
# Python is sometimes mistakenly considered thread-safe, however, this is not the
# case even with GIL enabled builds. There can still be data races between threads
# on pure Python objects.
#
# What is guaranteed is ref counts, memory management, etc being handled safely.
#
# For no-GIL builds, the interpreter lock is no longer in place to provide
# implicit safety for data that may be accessed by multiple threads when, say
# LineRequest.get_values is called
#
# The Python class wraps a PyObject (request_object) from the C extension that
# has buffers allocated at creation
#
# Calling get_values fills the buffer for that object instance which another thread
# could be writing/reading at the same time
#
# Without the GIL providing implicit synchronization, either the C extension or
# the caller are responsible for providing thread safety.
#
# As libgpiod is not marketed as being thread-safe, callers are ultimately
# responsible for using Chip or LineRequest objects shared by threads under the
# protection of a lock


def get_lock() -> "AbstractContextManager[None | bool]":
    """
    Helper function to return a lock that can return a nullcontext so that
    no lock is used. Can be used for a quick sanity check that things are not
    thread-safe
    """
    lock: AbstractContextManager[None | bool]
    if os.getenv("TESTS_NO_LOCKING"):
        lock = nullcontext()
        logger.debug("Running tests without locking")
    else:
        lock = threading.Lock()
    return lock


class ThreadedTestCase(TestCase):
    NUM_THREADS: ClassVar[int]
    ITERATIONS: ClassVar[int]
    TIMEOUT: ClassVar[int]

    def shortDescription(self) -> None:
        return None

    @classmethod
    def setUpClass(cls) -> None:
        cls.NUM_THREADS = 4
        # we want to stress test free threaded builds a bit more
        cls.ITERATIONS = 200 if is_free_threaded() else 20
        cls.TIMEOUT = 2


class Chip(ThreadedTestCase):
    def setUp(self) -> None:
        self.sim = gpiosim.Chip(
            num_lines=4, label="foobar", line_names={0: "l0", 1: "l1", 2: "l2", 3: "l3"}
        )
        self.chip = gpiod.Chip(self.sim.dev_path)

    def tearDown(self) -> None:
        self.chip.close()
        self.chip = None  # type: ignore[assignment]
        self.sim = None  # type: ignore[assignment]

    def test_per_thread_creation_and_query(self) -> None:
        """
        Test that multiple threads can create and query a chip pointing to the
        same backing device without a mutex

        LOCKING: Not required
        """

        barrier = threading.Barrier(self.NUM_THREADS, timeout=self.TIMEOUT)

        def worker(tid: int) -> None:
            barrier.wait()
            for _ in range(self.ITERATIONS):
                offset = tid % self.NUM_THREADS
                with gpiod.Chip(self.sim.dev_path) as chip:
                    info = chip.get_info()
                    self.assertEqual(
                        (info.name, info.label, info.num_lines),
                        (
                            self.sim.name,
                            "foobar",
                            4,
                        ),
                    )
                    line_info = chip.get_line_info(f"l{offset}")
                    self.assertEqual(
                        (line_info.offset, line_info.name), (offset, f"l{offset}")
                    )

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            futures = [executor.submit(worker, i) for i in range(self.NUM_THREADS)]
            for future in as_completed(futures, timeout=self.TIMEOUT):
                future.result(timeout=self.TIMEOUT)

    def test_shared_creation_and_query(self) -> None:
        """
        Test querying a single chip shared across multiple threads

        LOCKING: Not required
        """

        barrier = threading.Barrier(self.NUM_THREADS, timeout=self.TIMEOUT)
        lock = get_lock()

        def worker(tid: int) -> None:
            barrier.wait()
            for _ in range(self.ITERATIONS):
                offset = tid % self.NUM_THREADS
                with lock:
                    info = self.chip.get_info()
                self.assertEqual(
                    (info.name, info.label, info.num_lines),
                    (self.sim.name, "foobar", 4),
                )
                with lock:
                    line_info = self.chip.get_line_info(f"l{offset}")
                self.assertEqual(
                    (line_info.offset, line_info.name), (offset, f"l{offset}")
                )

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            futures = [executor.submit(worker, i) for i in range(self.NUM_THREADS)]
            for future in as_completed(futures, timeout=self.TIMEOUT):
                future.result(timeout=self.TIMEOUT)

    def test_shared_closed(self) -> None:
        """
        Tests that querying a single `Chip` shared across multiple threads after
        closing raises an error

        LOCKING: Required

        Note:
        The underlying `gpiod_chip` struct gets freed on close, leaving a mine
        for other threads to step on
        """

        barrier = threading.Barrier(self.NUM_THREADS, timeout=self.TIMEOUT)
        lock = get_lock()

        def worker() -> None:
            barrier.wait()
            with lock:
                info = self.chip.get_info()
                self.chip.close()
            self.assertEqual(
                (info.name, info.label, info.num_lines),
                (self.sim.name, "foobar", 4),
            )

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            futures = [executor.submit(worker) for _ in range(self.NUM_THREADS)]
            error_count = 0
            for future in as_completed(futures, timeout=self.TIMEOUT):
                try:
                    future.result(timeout=self.TIMEOUT)
                except gpiod.ChipClosedError:
                    error_count += 1
            self.assertEqual(error_count, self.NUM_THREADS - 1)


class InfoEvent(ThreadedTestCase):
    def setUp(self) -> None:
        self.sim = gpiosim.Chip(num_lines=4, label="foobar")
        self.chip = gpiod.Chip(self.sim.dev_path)

    def tearDown(self) -> None:
        self.chip.close()
        self.chip = None  # type: ignore[assignment]
        self.sim = None  # type: ignore[assignment]

    def test_watch_unwatch_line_info(self) -> None:
        """
        Tests that threads that share a `Chip` can watch/unwatch line info events

        LOCKING: Not strictly required

        Note:
        Threads may encounter EBUSY if the underlying file descriptor is busy or
        if the offset is already being watched
        """

        barrier = threading.Barrier(self.NUM_THREADS, timeout=self.TIMEOUT)
        num_lines = self.chip.get_info().num_lines

        def worker(tid: int) -> None:
            offset = tid % num_lines
            barrier.wait()
            for _ in range(self.ITERATIONS):
                try:
                    info = self.chip.watch_line_info(offset)
                    self.assertEqual(info.offset, offset)
                except OSError as e:
                    if e.errno == errno.EBUSY:
                        retry_count = 0
                        while retry_count < 2:
                            try:
                                retry_count += 1
                                self.chip.unwatch_line_info(offset)
                                break
                            except OSError as e:
                                pass

                info = self.chip.get_line_info(offset)
                self.assertEqual(info.offset, offset)

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            futures = [executor.submit(worker, _) for _ in range(self.NUM_THREADS)]
            for future in as_completed(futures, timeout=self.TIMEOUT):
                future.result(timeout=self.TIMEOUT)

    def test_watch_unwatch_line_info_locks(self) -> None:
        """
        Tests that threads that share a `Chip` can watch/unwatch line info events
        with locking

        Same as test_watch_unwatch_line_info but with locks and no EBUSY handling

        LOCKING: Not strictly required
        """

        barrier = threading.Barrier(self.NUM_THREADS, timeout=self.TIMEOUT)
        lock = get_lock()
        watching: set[int] = set()

        def worker(tid: int) -> None:
            barrier.wait()
            for _ in range(self.ITERATIONS):
                offset = tid % self.NUM_THREADS
                with lock:
                    if offset in watching:
                        self.chip.unwatch_line_info(offset)
                        watching.remove(offset)
                        info = self.chip.get_line_info(offset)
                    else:
                        info = self.chip.watch_line_info(offset)
                        watching.add(offset)
                self.assertEqual(info.offset, offset)

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            futures = [executor.submit(worker, i) for i in range(self.NUM_THREADS)]
            for future in as_completed(futures, timeout=self.TIMEOUT):
                future.result(timeout=self.TIMEOUT)

    def test_read_info_event(self) -> None:
        """
        Test that multiple threads that share a Chip can read info events

        LOCKING: Not required
        """

        num_lines = self.chip.get_info().num_lines
        for offset in range(num_lines):
            self.chip.watch_line_info(offset)
        # If read_edge_events() is blocking, threads will hang forever waiting
        # for events that don't exist when we're looking to shutdown.
        flags = fcntl.fcntl(self.chip.fd, fcntl.F_GETFL)
        fcntl.fcntl(self.chip.fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        worker_barrier = threading.Barrier(self.NUM_THREADS, timeout=self.TIMEOUT)
        feeder_barrier = threading.Barrier(2, timeout=self.TIMEOUT)
        done_fd = os.eventfd(0)

        total = 0
        counter_lock = threading.Lock()

        poll = epoll()
        poll.register(self.chip.fd, EPOLLIN)
        poll.register(done_fd, EPOLLIN)

        def reader_worker(tid: int) -> None:
            should_exit = False
            local_count = 0
            nonlocal total

            worker_barrier.wait()
            while not should_exit:
                events = poll.poll(timeout=self.TIMEOUT)

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
                                self.assertIsNotNone(_event)
                                local_count += 1
                        except OSError as e:
                            if e.errno == errno.EAGAIN:
                                continue
                            raise

            logger.debug(f"{tid} {local_count=}")
            with counter_lock:
                total += local_count

        def feeder(tid: int) -> None:
            offsets = list(range(tid, num_lines, 2))
            worker_barrier.wait()

            for i in range(int(self.ITERATIONS / 2)):
                offset = offsets[i % len(offsets)]
                with self.chip.request_lines(
                    config={offset: gpiod.LineSettings(direction=Direction.INPUT)}
                ) as req:
                    req.reconfigure_lines(
                        config={offset: gpiod.LineSettings(direction=Direction.OUTPUT)}
                    )

            feeder_barrier.wait()
            # Thread 0 signals done when all events have fired
            if tid == 0:
                os.eventfd_write(done_fd, 1)

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as ex:
            futures = [ex.submit(feeder, i) for i in range(2)]
            futures += [ex.submit(reader_worker, i) for i in range(2, self.NUM_THREADS)]

            try:
                for f in as_completed(futures, timeout=self.TIMEOUT):
                    f.result(timeout=self.TIMEOUT)
                self.assertGreater(total, 0)
            finally:
                for fd in [self.chip.fd, done_fd]:
                    poll.unregister(fd)
                poll.close()
                os.close(done_fd)
                for offset in range(num_lines):
                    self.chip.unwatch_line_info(offset)


class LineRequest(ThreadedTestCase):
    def setUp(self) -> None:
        self.sim = gpiosim.Chip(
            num_lines=4, label="foobar", line_names={0: "l0", 1: "l1", 2: "l2", 3: "l3"}
        )
        self.chip = gpiod.Chip(self.sim.dev_path)

    def tearDown(self) -> None:
        self.chip.close()
        self.chip = None  # type: ignore[assignment]
        self.sim = None  # type: ignore[assignment]

    def test_per_thread_creation_and_query(self) -> None:
        """
        Test that multiple threads can create and query their own LineRequest
        without a mutex

        LOCKING: Not strictly required

        Note: without a lock, EPERM may get raised due to the direction of the
        offset having been changed from output to input
        """

        barrier = threading.Barrier(self.NUM_THREADS, timeout=self.TIMEOUT)
        lock = get_lock()

        def worker(tid: int) -> None:
            # distribute threads across number of lines
            offset = 2 + (tid % 2)
            with lock:
                request = self.chip.request_lines(
                    config={offset: gpiod.LineSettings(direction=Direction.OUTPUT)}
                )
            counter = 0
            barrier.wait()
            for _ in range(self.ITERATIONS):
                try:
                    with lock:
                        direction = self.chip.get_line_info(offset).direction
                    if direction == Direction.INPUT:
                        continue
                    if request.get_value(offset) == Value.ACTIVE:
                        request.set_value(offset, Value.INACTIVE)
                        self.assertEqual(request.get_value(offset), Value.INACTIVE)
                        counter += 1
                    else:
                        request.set_value(offset, Value.ACTIVE)
                        self.assertEqual(request.get_value(offset), Value.ACTIVE)
                        counter += 1
                # set_value may raise a permission error when the pin is INPUT
                except OSError:
                    pass
            self.assertGreater(counter, 0)

        def feeder(tid: int) -> None:
            offset = tid % 2
            with lock:
                request = self.chip.request_lines(
                    config={offset: gpiod.LineSettings(direction=Direction.OUTPUT)}
                )
            barrier.wait()
            for iteration in range(self.ITERATIONS):
                new_dir = Direction.INPUT if iteration % 2 == 0 else Direction.OUTPUT
                with lock:
                    request.reconfigure_lines(
                        config={offset: gpiod.LineSettings(direction=new_dir)}
                    )

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            futures = [executor.submit(feeder, i) for i in range(2)]
            futures += [executor.submit(worker, i) for i in range(2, self.NUM_THREADS)]
            for future in as_completed(futures, timeout=self.TIMEOUT):
                future.result(timeout=self.TIMEOUT)

    def test_shared_creation_and_query(self) -> None:
        """
        Test multiple threads can reconfigure, set values and get values on a
        shared line request

        LOCKING: Required

        Note:
        This won't actually blow up, but based on the extension implementation
        the request has a shared buffer for offets/values that are reused for
        getting/setting line values

        Without synchronization, a thread may think it's setting one set of values
        but the buffer values may have been overwritten by another thread

        Implementation Note:
        We use a dual set of events to make sure the feeder/worker pair alternate
        otherwise a thread may monopolize the lock and finish before triggering
        a set_value call. We pair this with a lock to prevent issues with the
        aforementioned buffer contention.
        """

        barrier = threading.Barrier(self.NUM_THREADS, timeout=self.TIMEOUT)
        lock = get_lock()
        num_lines = self.chip.get_info().num_lines
        request = self.chip.request_lines(
            config={range(num_lines): gpiod.LineSettings(direction=Direction.OUTPUT)}
        )

        events_ready = {0: threading.Event(), 1: threading.Event()}
        events_set = {0: threading.Event(), 1: threading.Event()}

        def worker(tid: int) -> None:
            # we're using 2 feeder threads, each with a dedicated offset
            offset = tid % 2
            counter = 0
            _ready = events_ready[offset]
            _set = events_set[offset]
            _set.set()
            barrier.wait()
            for _ in range(self.ITERATIONS):
                _ready.wait(self.TIMEOUT)
                _ready.clear()
                with lock:
                    if self.chip.get_line_info(offset).direction == Direction.OUTPUT:
                        if request.get_value(offset) == Value.ACTIVE:
                            request.set_value(offset, Value.INACTIVE)
                            self.assertEqual(request.get_value(offset), Value.INACTIVE)
                            counter += 1
                        else:
                            request.set_value(offset, Value.ACTIVE)
                            self.assertEqual(request.get_value(offset), Value.ACTIVE)
                            counter += 1
                _set.set()
            self.assertGreater(counter, 0)

        def feeder(tid: int) -> None:
            offset = tid % 2
            _ready = events_ready[offset]
            _set = events_set[offset]
            barrier.wait()
            for iteration in range(self.ITERATIONS):
                new_dir = Direction.INPUT if iteration % 2 == 0 else Direction.OUTPUT
                _set.wait(self.TIMEOUT)
                _set.clear()
                with lock:
                    request.reconfigure_lines(
                        config={offset: gpiod.LineSettings(direction=new_dir)}
                    )
                _ready.set()

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            futures = [executor.submit(feeder, i) for i in range(2)]
            futures += [executor.submit(worker, i) for i in range(2, self.NUM_THREADS)]
            try:
                for future in as_completed(futures, timeout=self.TIMEOUT):
                    future.result(timeout=self.TIMEOUT)
            finally:
                request.release()

    def test_shared_set_get_values(self) -> None:
        """
        Test setting and getting values from a single line request shared across
        multiple threads

        LOCKING: Required

        Note:
        This won't actually blow up, but based on the extension implementation
        the request has a shared buffer for offets/values that are reused for
        getting/setting line values

        Without synchronization, a thread may think it's setting one set of values
        but the buffer values may have been overwritten by another thread
        """

        barrier = threading.Barrier(self.NUM_THREADS, timeout=self.TIMEOUT)
        lock = get_lock()
        offset = 0
        request = self.chip.request_lines(
            config={0: gpiod.LineSettings(direction=Direction.OUTPUT)}
        )

        def worker() -> None:
            counter = 0
            barrier.wait()
            for _ in range(self.ITERATIONS):
                with lock:
                    if request.get_value(offset) == Value.ACTIVE:
                        request.set_value(offset, Value.INACTIVE)
                        self.assertEqual(request.get_value(offset), Value.INACTIVE)
                        counter += 1
                    else:
                        request.set_value(offset, Value.ACTIVE)
                        self.assertEqual(request.get_value(offset), Value.ACTIVE)
                        counter += 1
            self.assertGreater(counter, 0)

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            futures = [executor.submit(worker) for _ in range(self.NUM_THREADS)]
            try:
                for future in as_completed(futures, timeout=self.TIMEOUT):
                    future.result(timeout=self.TIMEOUT)
            finally:
                request.release()

    def test_shared_close(self) -> None:
        """
        Test that querying a single line request shared across multiple threads
        after releasing raises an error

        LOCKING: Required

        Note:
        The underlying `gpiod_line_request` struct gets freed on release, leaving
        a mine for other threads to step on
        """
        barrier = threading.Barrier(self.NUM_THREADS, timeout=self.TIMEOUT)
        lock = get_lock()

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
                self.assertEqual(line, Value.INACTIVE)

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as executor:
            futures = [executor.submit(worker) for _ in range(self.NUM_THREADS)]
            error_count = 0
            for future in as_completed(futures, timeout=self.TIMEOUT):
                try:
                    future.result(timeout=self.TIMEOUT)
                except gpiod.RequestReleasedError:
                    error_count += 1
            self.assertEqual(error_count, self.NUM_THREADS - 1)


class EdgeEvent(ThreadedTestCase):
    def setUp(self) -> None:
        self.sim = gpiosim.Chip(num_lines=4, label="foobar")
        self.chip = gpiod.Chip(self.sim.dev_path)

    def tearDown(self) -> None:
        self.chip.close()
        self.sim = None  # type: ignore[assignment]
        self.chip = None  # type: ignore[assignment]

    def test_read_edge_events(self) -> None:
        """
        Test that multiple threads can read edge events on a shared LineRequest

        LOCKING: Required

        Note:
        The request object has a gpiod_edge_event_buffer for events to be read into.
        Without synchronization, that buffer will be overwritten by another thread
        when attempting to create event objects
        """
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

        worker_barrier = threading.Barrier(self.NUM_THREADS, timeout=self.TIMEOUT)
        feeder_barrier = threading.Barrier(2, timeout=self.TIMEOUT)
        done_fd = os.eventfd(0)

        total = 0
        counter_lock = threading.Lock()
        req_lock = get_lock()

        poll = epoll()
        poll.register(req.fd, EPOLLIN)
        poll.register(done_fd, EPOLLIN)

        def reader_worker(tid: int) -> None:
            nonlocal total
            should_exit = False
            local_count = 0
            worker_barrier.wait()

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
                        except OSError as e:
                            if e.errno == errno.EAGAIN:
                                continue
                            raise

            logger.debug(f"{tid} {local_count=}")
            with counter_lock:
                total += local_count

        def feeder(tid: int) -> None:
            offsets = list(range(tid, num_lines, 2))
            worker_barrier.wait()

            for i in range(int(self.ITERATIONS / 2)):
                offset = offsets[i % len(offsets)]
                for pull in [gpiosim.Chip.Pull.UP, gpiosim.Chip.Pull.DOWN]:
                    self.sim.set_pull(offset, pull)

            feeder_barrier.wait()
            # Thread 0 signals done when all pulses have fired
            if tid == 0:
                os.eventfd_write(done_fd, 1)

        with ThreadPoolExecutor(max_workers=self.NUM_THREADS) as ex:
            futures = [ex.submit(feeder, i) for i in range(2)]
            futures += [ex.submit(reader_worker, i) for i in range(2, self.NUM_THREADS)]

            try:
                for f in as_completed(futures, timeout=self.TIMEOUT):
                    f.result(timeout=self.TIMEOUT)
                self.assertGreater(total, 0)
            finally:
                for fd in [req.fd, done_fd]:
                    poll.unregister(fd)
                poll.close()
                os.close(done_fd)
                req.release()
