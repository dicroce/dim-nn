
#ifndef __mm_h
#define __mm_h

#include <utility>
#include <sys/mman.h>
#include <stdexcept>
#include <cstdint>
#include <cstdlib>

static const uint32_t MAX_MAPPING_LEN = 1048576000;

class r_memory_map
{
public:
    enum Flags
    {
        RMM_TYPE_FILE = 0x01,
        RMM_TYPE_ANON = 0x02,
        RMM_SHARED = 0x04,
        RMM_PRIVATE = 0x08,
        RMM_FIXED = 0x10
    };

    enum Protection
    {
        RMM_PROT_NONE = 0x00,
        RMM_PROT_READ = 0x01,
        RMM_PROT_WRITE = 0x02,
        RMM_PROT_EXEC = 0x04
    };

    enum Advice
    {
        RMM_ADVICE_NORMAL = 0x00,
        RMM_ADVICE_RANDOM = 0x01,
        RMM_ADVICE_SEQUENTIAL = 0x02,
        RMM_ADVICE_WILLNEED = 0x04,
        RMM_ADVICE_DONTNEED = 0x08
    };

    r_memory_map() :
        _mem(nullptr),
        _length(0)
    {
    }

    r_memory_map(
        int fd,
        int64_t offset,
        uint32_t len,
        uint32_t prot,
        uint32_t flags
    ) :
        _mem(nullptr),
        _length(len)
    {
        if( fd <= 0 )
            throw std::runtime_error( "Attempting to memory map a bad file descriptor." );

        if( (len == 0) || (len > MAX_MAPPING_LEN) )
            throw std::runtime_error( "Attempting to memory map more than 1gb is invalid." );

        if( !(flags & RMM_TYPE_FILE) && !(flags & RMM_TYPE_ANON) )
            throw std::runtime_error( "A mapping must be either a file mapping, or an anonymous mapping (neither was specified)." );

        if( flags & RMM_FIXED )
            throw std::runtime_error( "r_memory_map does not support fixed mappings." );

        _mem = mmap( NULL,
                    _length,
                    _GetPosixProtFlags( prot ),
                    _GetPosixAccessFlags( flags ),
                    fd,
                    offset );

        if(_mem == MAP_FAILED)
            throw std::runtime_error( "Unable to complete file mapping");
    }
    
    r_memory_map(const r_memory_map&) = delete;

    r_memory_map(r_memory_map&& other) :
        _mem(std::move(other._mem)),
        _length(std::move(other._length))
    {
        other._mem = nullptr;
        other._length = 0;
    }

    virtual ~r_memory_map() noexcept
    {
        _clear();
    }

    r_memory_map& operator=(const r_memory_map& other) = delete;

    r_memory_map& operator=(r_memory_map&& other) noexcept
    {
        if(this != &other)
        {
            _clear();

            _mem = std::move(other._mem);
            other._mem = nullptr;
            _length = std::move(other._length);
            other._length = 0;
        }

        return *this;
    }

    inline void* map() const
    {
        return _mem;
    }

    inline uint32_t length() const
    {
        return _length;
    }

    inline bool mapped() const
    {
        return _mem != nullptr;
    }

    void advise(int advice, void* addr = nullptr, size_t length = 0) const
    {
        int posixAdvice = _GetPosixAdvice( advice );

        int err = madvise( (addr)?addr:_mem, (length>0)?length:_length, posixAdvice );

        if( err != 0 )
            throw std::runtime_error( "Unable to apply memory mapping advice." );
    }

    void flush(void* addr = nullptr, size_t length = 0, bool now = true)
    {
        int err = msync( (addr)?addr:_mem, (length>0)?length:_length, (now) ? MS_SYNC : MS_ASYNC );

        if( err != 0 )
            throw std::runtime_error("Unable to sync memory mapped file.");
    }

private:
    void _clear() noexcept
    {
        if(_mem != nullptr)
        {
            munmap( _mem, _length );
            _mem = nullptr;
        }
    }

    int _GetPosixProtFlags( int prot ) const
    {
        int osProtFlags = 0;

        if( prot & RMM_PROT_READ )
            osProtFlags |= PROT_READ;
        if( prot & RMM_PROT_WRITE )
            osProtFlags |= PROT_WRITE;
        if( prot & RMM_PROT_EXEC )
            osProtFlags |= PROT_EXEC;

        return osProtFlags;
    }
    int _GetPosixAccessFlags( int flags ) const
    {
        int osFlags = 0;

        if( flags & RMM_TYPE_FILE )
            osFlags |= MAP_FILE;
        if( flags & RMM_TYPE_ANON )
            osFlags |= MAP_ANONYMOUS;
        if( flags & RMM_SHARED )
            osFlags |= MAP_SHARED;
        if( flags & RMM_PRIVATE )
            osFlags |= MAP_PRIVATE;
        if( flags & RMM_FIXED )
            osFlags |= MAP_FIXED;

        return osFlags;
    }
    int _GetPosixAdvice( int advice ) const
    {
        int posixAdvice = 0;

        if( advice & RMM_ADVICE_RANDOM )
            posixAdvice |= MADV_RANDOM;
        if( advice & RMM_ADVICE_SEQUENTIAL )
            posixAdvice |= MADV_SEQUENTIAL;
        if( advice & RMM_ADVICE_WILLNEED )
            posixAdvice |= MADV_WILLNEED;
        if( advice & RMM_ADVICE_DONTNEED )
            posixAdvice |= MADV_DONTNEED;

        return posixAdvice;
    }

    void* _mem;
    uint32_t _length;
};

class fil
{
public:
    fil(const std::string& name, const std::string& mode) :
        _f(fopen(name.c_str(), mode.c_str()))
    {
    }
    ~fil()
    {
        if(_f)
            fclose(_f);
    }

    int fd()
    {
        return fileno(_f);
    }
private:
    FILE* _f;
};

#endif