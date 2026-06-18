// ============================================================================
// syscalls.cpp
//
// THE PROBLEM:
//   riscv64-unknown-elf-g++ links against "newlib", a tiny C library built
//   for bare-metal targets (no OS underneath). Newlib does NOT know how to
//   read a file, write to stdout, or grow the heap. It expects YOU to supply
//   a handful of low-level functions (the "_read", "_write", "_open", etc.
//   family) that actually talk to whatever hardware/OS is really there.
//
//   qemu-riscv64 (the *user-mode* emulator you are using) is not a bare-metal
//   board. It emulates a Linux process: it expects the program to issue real
//   Linux syscalls via the `ecall` instruction (the RISC-V equivalent of a
//   software interrupt / trap into the kernel), using the standard Linux
//   RV64 syscall calling convention:
//
//       a7 = syscall number
//       a0..a5 = syscall arguments
//       ecall
//       a0 = return value (or -errno on failure)
//
// THE FIX (this file):
//   We implement newlib's missing stub functions ourselves. Each one is a
//   tiny "translator": it takes the abstract newlib request ("write these
//   bytes", "read this much", "give me more heap") and turns it into the
//   exact Linux syscall QEMU is listening for. That's the whole concept —
//   this file IS the bridge between your C++ program and QEMU.
//
// NOTE:
//   Because newlib's prototypes for these are C (not C++), we declare them
//   inside extern "C" so the linker can actually find them when newlib's
//   internals (written in C) call _write(), _read(), etc.
// ============================================================================

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cerrno>
#include <sys/stat.h>
#include <sys/types.h>

extern "C" {

#define SYS_read    63
#define SYS_write   64
#define SYS_openat  56
#define SYS_close   57
#define SYS_lseek   62
#define SYS_exit    93
#define SYS_brk     214

#define AT_FDCWD    -100

static inline long raw_syscall(long n, long a0, long a1, long a2, long a3) {
    register long a7 asm("a7") = n;
    register long r0 asm("a0") = a0;
    register long r1 asm("a1") = a1;
    register long r2 asm("a2") = a2;
    register long r3 asm("a3") = a3;
    asm volatile("ecall"
                 : "+r"(r0)
                 : "r"(a7), "r"(r1), "r"(r2), "r"(r3)
                 : "memory");
    return r0;
}

int _write(int fd, const char *buf, int len) {
    long ret = raw_syscall(SYS_write, fd, (long)buf, len, 0);
    if (ret < 0) { errno = (int)-ret; return -1; }
    return (int)ret;
}

int _read(int fd, char *buf, int len) {
    long ret = raw_syscall(SYS_read, fd, (long)buf, len, 0);
    if (ret < 0) { errno = (int)-ret; return -1; }
    return (int)ret;
}

#define NEWLIB_O_WRONLY   0x0001
#define NEWLIB_O_RDWR     0x0002
#define NEWLIB_O_CREAT    0x0200
#define NEWLIB_O_TRUNC    0x0400
#define NEWLIB_O_APPEND   0x0008

#define LINUX_O_WRONLY    0x0001
#define LINUX_O_RDWR      0x0002
#define LINUX_O_CREAT     0x0040
#define LINUX_O_TRUNC     0x0200
#define LINUX_O_APPEND    0x0400

static int translate_open_flags(int newlib_flags) {
    int linux_flags = 0;
    if (newlib_flags & NEWLIB_O_WRONLY) linux_flags |= LINUX_O_WRONLY;
    if (newlib_flags & NEWLIB_O_RDWR)   linux_flags |= LINUX_O_RDWR;
    if (newlib_flags & NEWLIB_O_CREAT)  linux_flags |= LINUX_O_CREAT;
    if (newlib_flags & NEWLIB_O_TRUNC)  linux_flags |= LINUX_O_TRUNC;
    if (newlib_flags & NEWLIB_O_APPEND) linux_flags |= LINUX_O_APPEND;
    return linux_flags;
}

int _open(const char *name, int flags, int mode) {
    int linux_flags = translate_open_flags(flags);
    long ret = raw_syscall(SYS_openat, AT_FDCWD, (long)name, linux_flags, mode);
    if (ret < 0) { errno = (int)-ret; return -1; }
    return (int)ret;
}

int _close(int fd) {
    long ret = raw_syscall(SYS_close, fd, 0, 0, 0);
    if (ret < 0) { errno = (int)-ret; return -1; }
    return (int)ret;
}

long _lseek(int fd, long offset, int whence) {
    long ret = raw_syscall(SYS_lseek, fd, offset, whence, 0);
    if (ret < 0) { errno = (int)-ret; return -1; }
    return ret;
}

int _fstat(int fd, struct stat *st) {
    (void)fd;
    if (st) {
        memset(st, 0, sizeof(struct stat));
        st->st_mode = S_IFREG | 0644;
        st->st_blksize = 512;
    }
    return 0;
}

int _isatty(int fd) {
    (void)fd;
    return 0;
}

void* _sbrk(ptrdiff_t increment) {
    static uint8_t* heap_end = nullptr;
    if (heap_end == nullptr) {
        heap_end = (uint8_t*)raw_syscall(SYS_brk, 0, 0, 0, 0);
    }
    uint8_t* prev_end = heap_end;
    uint8_t* new_end   = heap_end + increment;
    long ret = raw_syscall(SYS_brk, (long)new_end, 0, 0, 0);
    if ((uint8_t*)ret < new_end) {
        errno = ENOMEM;
        return (void*)-1;
    }
    heap_end = new_end;
    return prev_end;
}

void _exit(int status) {
    raw_syscall(SYS_exit, status, 0, 0, 0);
    while (1) {}
}

int _kill(int pid, int sig) { (void)pid; (void)sig; errno = ENOSYS; return -1; }
int _getpid(void) { return 1; }

} // extern "C"
