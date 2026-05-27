#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
enum class FloatEncoding {
    E4M3,
    E5M2
};

template<FloatEncoding T>
struct alignas(1) float8_base {

    static constexpr bool IS_E4M3 = (T == FloatEncoding::E4M3);
    static constexpr bool IS_E5M2 = (T == FloatEncoding::E5M2);

    // Number of Bits representing mantissa and exponents
    static constexpr int FP32_NUM_BITS = 32;
    static constexpr int FP32_NUM_EXPONENT_BITS = 8;
    static constexpr int FP32_NUM_MANTISSA_BITS = 23;
    static constexpr uint32_t FP32_NAN = 0x7fffffff;
    static constexpr uint32_t FP32_INFINITY_MASK = 0x7f800000;
    static constexpr int FP32_MAX_EXPONENT  =  127;
    static constexpr int FP32_MIN_EXPONENT  = -126;
    static constexpr int FP32_EXPONENT_BIAS =  127;

    static constexpr int FP16_NUM_BITS = 16;
    static constexpr int FP16_NUM_EXPONENT_BITS = 5;
    static constexpr int FP16_NUM_MANTISSA_BITS = 10;
    static constexpr uint16_t FP16_NAN = 0x7fff;
    static constexpr uint16_t FP16_INFINITY_MASK = 0x7c00;
    static constexpr int FP16_MAX_EXPONENT  = 15;
    static constexpr int FP16_MIN_EXPONENT  = -14;
    static constexpr int FP16_EXPONENT_BIAS = 15;

    static constexpr int FP8_NUM_BITS = 8;
    static constexpr int FP8_NUM_EXPONENT_BITS = IS_E4M3 ? 4 : 5;
    static constexpr int FP8_NUM_MANTISSA_BITS = IS_E4M3 ? 3 : 2;
    static constexpr uint8_t  FP8_NAN = 0x7f; // Also F8_INF
    static constexpr uint8_t  FP8_INFINITY_MASK = IS_E4M3 ? 0x78 : 0x7c;
    static constexpr int FP8_MAX_EXPONENT  = IS_E4M3 ?  7 :  15;
    static constexpr int FP8_MIN_EXPONENT  = IS_E4M3 ? -6 : -14;
    static constexpr int FP8_EXPONENT_BIAS = IS_E4M3 ?  7 :  15;

    static constexpr uint8_t  FP8_EXPONENT_MASK = (1 << FP8_NUM_EXPONENT_BITS) - 1;
    static constexpr uint8_t  FP8_MANTISSA_MASK = (1 << FP8_NUM_MANTISSA_BITS) - 1;

    static constexpr uint8_t FP8_MAX_FLT = (IS_E4M3 ? 0x7e : 0x7b);

    // 256 in float
    static constexpr uint32_t FP8_SAT_VAL_FP32 = 0x43800000;

    //
    // Data members
    //

    /// Data container
    uint8_t storage;

    /// Ctors.

    float8_base() : storage(0) { }

    /// Is finite implementation

    static bool isfinite(float flt) {
        uint32_t s;

        #if defined(__CUDA_ARCH__)
        s = reinterpret_cast<uint32_t const &>(flt);
        #else
        std::memcpy(&s, &flt, sizeof(s));
        #endif

        return (s & 0x7f800000) < 0x7f800000;
    }

    /// Is NaN implementation

    static bool isnan(float flt) {
        uint32_t s;

        #if defined(__CUDA_ARCH__)
        s = reinterpret_cast<uint32_t const &>(flt);
        #else
        std::memcpy(&s, &flt, sizeof(s));
        #endif

        return (s & 0x7fffffff) > 0x7f800000;
    }

    /// Is infinite implementation

    static bool isinf(float flt) {
        uint32_t s;

        #if defined(__CUDA_ARCH__)
        s = reinterpret_cast<uint32_t const &>(flt);
        #else
        std::memcpy(&s, &flt, sizeof(s));
        #endif

        // Sign = 0 for +inf, 1 for -inf
        // Exponent = all ones
        // Mantissa = all zeros
        return (s == 0x7f800000) || (s == 0xff800000);
    }

    /// FP32 -> FP8 conversion - rounds to nearest even

    static uint8_t convert_float_to_fp8(float const& flt) {

        // software implementation rounds toward nearest even
        uint32_t s;

        #if defined(__CUDA_ARCH__)
        s = reinterpret_cast<uint32_t const &>(flt);
        #else
        std::memcpy(&s, &flt, sizeof(s));
        #endif

        // Extract the bits in the FP32 type
        uint8_t sign = uint8_t((s >> 24 & 0x80));
        int32_t exp = int32_t((s >> FP32_NUM_MANTISSA_BITS) & 0xff) - FP32_EXPONENT_BIAS;
        int mantissa = s & 0x7fffff;
        uint8_t u = 0;

        uint8_t const kF8_NaN = 0x7f;

        // NaN => NaN
        if (isnan(flt)) {
            return kF8_NaN;
        }

        // Inf => MAX_FLT (satfinite)
        if (isinf(flt)) {
            return sign | FP8_MAX_FLT;
        }

        // Special handling
        if (exp == -128) {
            // int8 range is from -128 to 127
            // So 255(inf) - 127(bias) = 128 - will show up as -128

            // satfinite
            return (sign | FP8_MAX_FLT);
        }

        int sticky_bit = 0;

        bool skip_sign = false;
        bool may_be_nan = false;

        if ( (exp >= FP8_MIN_EXPONENT) && (exp <= FP8_MAX_EXPONENT) ) {
            // normal fp32 to normal fp8
            exp = exp + FP8_EXPONENT_BIAS;
            u = uint8_t((uint32_t(exp) & FP8_EXPONENT_MASK) << FP8_NUM_MANTISSA_BITS);
            u = uint8_t(u | (mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)));
        } else if(exp < FP8_MIN_EXPONENT) {
            // normal single-precision to subnormal float8-precision representation
            int rshift = (FP8_MIN_EXPONENT - exp);
            if (rshift < FP32_NUM_BITS) {
                mantissa |= (1 << FP32_NUM_MANTISSA_BITS);

                sticky_bit = ((mantissa & ((1 << rshift) - 1)) != 0);

                mantissa = (mantissa >> rshift);
                u = (uint8_t(mantissa >> (FP32_NUM_MANTISSA_BITS- FP8_NUM_MANTISSA_BITS)) & FP8_MANTISSA_MASK);
            } else {
                mantissa = 0;
                u = 0;
            }
        // Exponent > FP8_MAX_EXPONENT - this is a special case done to match HW
        // 0x4380_0000 to 0x43e0_0000 - maps from 256 to 448, and does not saturate / inf.
        } else {
            if( exp == (FP8_MAX_EXPONENT + 1) ) {
                uint8_t mantissa_tmp = uint8_t(mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS));
                if( mantissa_tmp < FP8_MANTISSA_MASK) {
                    exp = exp + FP8_EXPONENT_BIAS;
                    u = uint8_t(uint32_t(exp) << FP8_NUM_MANTISSA_BITS) | mantissa_tmp;
                    may_be_nan =  (mantissa_tmp == (FP8_MANTISSA_MASK-1));
                } else {
                    // satfinite
                    return (sign | FP8_MAX_FLT);
                }
            } else{
                // satfinite
                return (sign | FP8_MAX_FLT);
            }
        }

        // round to nearest even
        int NUM_BITS_SHIFT = FP32_NUM_MANTISSA_BITS - (FP8_NUM_MANTISSA_BITS + 1);
        int round_bit = ((mantissa >> NUM_BITS_SHIFT) & 1);
        sticky_bit |= ((mantissa & ((1 << NUM_BITS_SHIFT) - 1)) != 0);

        if ((round_bit && sticky_bit) || (round_bit && (u & 1))) {
            u = uint8_t(u + 1);
            if( may_be_nan ) {
                skip_sign = true;
            }
        }

        if (u > FP8_MAX_FLT) {
            // satfinite
            u = (sign | FP8_MAX_FLT);
        }

        if( ! skip_sign ) {
            u |= sign;
        }

        return u;
    }


    /// Converts a fp8 value stored as a uint8_t to a float

    static float convert_fp8_to_float(uint8_t const& x) {

        uint32_t constexpr kF32_NaN = 0x7fffffff;

        uint8_t const &f8 = x;
        uint32_t sign = (f8 >> (FP8_NUM_BITS - 1)) & 1;
        uint32_t exp = (f8 >> FP8_NUM_MANTISSA_BITS) & FP8_EXPONENT_MASK;
        uint32_t mantissa = f8 & FP8_MANTISSA_MASK;
        unsigned f = (sign << (FP32_NUM_BITS-1));

        if (IS_E4M3 && exp == 15 && mantissa == 0x7) {
            f = kF32_NaN;
        }
        else if (exp > 0 && (IS_E4M3 || exp < (FP8_MAX_EXPONENT + FP8_EXPONENT_BIAS + 1))) {
            // normal
            exp += (FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS);
            f = f |
                (exp << FP32_NUM_MANTISSA_BITS) |
                (mantissa << (FP32_NUM_MANTISSA_BITS-FP8_NUM_MANTISSA_BITS));
        } else if (exp == 0) {
            if (mantissa) {
                // subnormal
                exp += (FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS) + 1;
                while ((mantissa & (1 << FP8_NUM_MANTISSA_BITS)) == 0) {
                    mantissa <<= 1;
                    exp--;
                }
                mantissa &= FP8_MANTISSA_MASK;
                f = f |
                    (exp << FP32_NUM_MANTISSA_BITS) |
                    (mantissa << (FP32_NUM_MANTISSA_BITS-FP8_NUM_MANTISSA_BITS));
            } else {
                // sign-preserving zero
            }
        } else {
            if(mantissa == 0){
                // Sign-preserving infinity
                f = (f | 0x7f800000);
            } else {
                // Canonical NaN
                f = kF32_NaN;
            }
        }

        #if defined(__CUDA_ARCH__)
        return reinterpret_cast<float const&>(f);
        #else
        float flt;
        std::memcpy(&flt, &f, sizeof(flt));
        return flt;
        #endif
    }
};

///////////////////////////////////////////////////////////////
///
/// floating-point 8 type : E4M3
///
///////////////////////////////////////////////////////////////
struct alignas(1) float_e4m3_t : float8_base<FloatEncoding::E4M3> {

    using Base = float8_base<FloatEncoding::E4M3>;

    static constexpr int MAX_EXPONENT = Base::FP8_MAX_EXPONENT;

    //
    // Static conversion operators
    //

    /// Constructs from an uint8_t

    static float_e4m3_t bitcast(uint8_t x) {
        float_e4m3_t f;
        f.storage = x;
        return f;
    }

    /// FP32 -> FP8 conversion - rounds to nearest even

    static float_e4m3_t from_float(float const& flt) {
    #if defined(CUDA_PTX_FP8_CVT_ENABLED)
        uint16_t tmp;
        float y = float();
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(tmp) : "f"(y), "f"(flt));

        return *reinterpret_cast<float_e4m3_t *>(&tmp);
    #else
        return bitcast(Base::convert_float_to_fp8(flt));
    #endif
    }

    // E4M3 -> Float

    static float to_float(float_e4m3_t const& x) {
    #if defined(CUDA_PTX_FP8_CVT_ENABLED)
        uint16_t bits = x.storage;
        uint32_t packed;
        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(packed) : "h"(bits));

        return __half2float(reinterpret_cast<half2 const &>(packed).x);
    #else
        return Base::convert_fp8_to_float(x.storage);
    #endif
    }

    //
    // Methods
    //

    /// Constructor inheritance
    using Base::Base;

    /// Default constructor
    float_e4m3_t() = default;

#ifdef CUDA_FP8_ENABLED
    /// Conversion from CUDA's FP8 type

    explicit float_e4m3_t(__nv_fp8_e4m3 x) {
        storage = x.__x;
    }
#endif

    /// Floating point conversion

    explicit float_e4m3_t(float x) {
        storage = from_float(x).storage;
    }

    /// Floating point conversion

    explicit float_e4m3_t(double x): float_e4m3_t(float(x)) {
    }

    /// Integer conversion

    explicit float_e4m3_t(int x): float_e4m3_t(float(x)) {
    }


    explicit float_e4m3_t(unsigned x): float_e4m3_t(float(x)) {
    }

#ifdef CUDA_FP8_ENABLED
    /// Assignment from CUDA's FP8 type

    float_e4m3_t & operator=(__nv_fp8_e4m3 x) {
        storage = x.__x;
        return *this;
    }
#endif

    /// Converts to float

    operator float() const {
        return to_float(*this);
    }


    explicit operator double() const {
        return double(to_float(*this));
    }

    /// Converts to int

    explicit operator int() const {
    #if defined(__CUDA_ARCH__)
        return __half2int_rn(to_half(*this));
    #else
        return int(to_float(*this));
    #endif
    }

    /// Casts to bool

    explicit operator bool() const {
    #if defined(__CUDA_ARCH__)
        return bool(__half2int_rn(to_half(*this)));
    #else
        return bool(int(to_float(*this)));
    #endif
    }

    /// Accesses raw internal state

    uint8_t& raw() {
        return storage;
    }

    /// Accesses raw internal state

    uint8_t raw() const {
        return storage;
    }

    /// Returns the sign bit

    bool signbit() const {
        return ((storage & (1 << (Base::FP8_NUM_BITS - 1))) != 0);
    }

    /// Returns the biased exponent

    int exponent_biased() const {
        return int((storage >> FP8_NUM_MANTISSA_BITS) & Base::FP8_EXPONENT_MASK);
    }

    /// Returns the unbiased exponent

    int exponent() const {
        return exponent_biased() - 15;
    }

    /// Returns the mantissa

    int mantissa() const {
        return int(storage & Base::FP8_MANTISSA_MASK);
    }


    friend bool isnan(float_e4m3_t const& x) {
      return x.storage == uint8_t(0x7f);
    }
};

struct alignas(1) float_e5m2_t : float8_base<FloatEncoding::E5M2> {

    using Base = float8_base<FloatEncoding::E5M2>;

    static constexpr int MAX_EXPONENT = Base::FP8_MAX_EXPONENT;

    //
    // Static conversion operators
    //

    /// Constructs from an uint8_t

    static float_e5m2_t bitcast(uint8_t x) {
        float_e5m2_t f;
        f.storage = x;
        return f;
    }

    /// FP32 -> FP8 conversion - rounds to nearest even

    static float_e5m2_t from_float(float const& flt) {
    #if defined(CUDA_PTX_FP8_CVT_ENABLED)
        uint16_t tmp;
        float y = float();
        asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(tmp) : "f"(y), "f"(flt));

        return *reinterpret_cast<float_e5m2_t *>(&tmp);
    #else
        return bitcast(Base::convert_float_to_fp8(flt));
    #endif
    }

    /// FP16 -> E5M2 conversion - rounds to nearest even

    // E5M2 -> Float

    static float to_float(float_e5m2_t const& x) {
    #if defined(CUDA_PTX_FP8_CVT_ENABLED)
        uint16_t bits = x.storage;
        uint32_t packed;
        asm volatile("cvt.rn.f16x2.e5m2x2 %0, %1;\n" : "=r"(packed) : "h"(bits));

        return __half2float(reinterpret_cast<half2 const &>(packed).x);
    #else
        return Base::convert_fp8_to_float(x.storage);
    #endif
    }

    //
    // Methods
    //

    /// Constructor inheritance
    using Base::Base;

    /// Default constructor
    float_e5m2_t() = default;

#ifdef CUDA_FP8_ENABLED
    /// Conversion from CUDA's FP8 type

    explicit float_e5m2_t(__nv_fp8_e5m2 x) {
        storage = x.__x;
    }
#endif

    /// Floating point conversion

    explicit float_e5m2_t(float x) {
        storage = from_float(x).storage;
    }



    /// Floating point conversion

    explicit float_e5m2_t(double x): float_e5m2_t(float(x)) {
    }

    /// Integer conversion

    explicit float_e5m2_t(int x): float_e5m2_t(float(x)) {
    }


    explicit float_e5m2_t(unsigned x): float_e5m2_t(float(x)) {
    }

    /// E4M3 conversion

    explicit float_e5m2_t(float_e4m3_t x);

#ifdef CUDA_FP8_ENABLED
    /// Assignment from CUDA's FP8 type

    float_e5m2_t & operator=(__nv_fp8_e5m2 x) {
        storage = x.__x;
        return *this;
    }
#endif

    /// Converts to float

    operator float() const {
        return to_float(*this);
    }

    /// Converts to float

    explicit operator double() const {
        return double(to_float(*this));
    }

    /// Converts to int

    explicit operator int() const {
    #if defined(__CUDA_ARCH__)
        return __half2int_rn(to_half(*this));
    #else
        return int(to_float(*this));
    #endif
    }

    /// Casts to bool

    explicit operator bool() const {
    #if defined(__CUDA_ARCH__)
        return bool(__half2int_rn(to_half(*this)));
    #else
        return bool(int(to_float(*this)));
    #endif
    }

    /// Accesses raw internal state

    uint8_t& raw() {
        return storage;
    }

    /// Accesses raw internal state

    uint8_t raw() const {
        return storage;
    }

    /// Returns the sign bit

    bool signbit() const {
        return ((storage & (1 << (Base::FP8_NUM_BITS - 1))) != 0);
    }

    /// Returns the biased exponent

    int exponent_biased() const {
        return int((storage >> FP8_NUM_MANTISSA_BITS) & Base::FP8_EXPONENT_MASK);
    }

    /// Returns the unbiased exponent

    int exponent() const {
        return exponent_biased() - 15;
    }

    /// Returns the mantissa

    int mantissa() const {
        return int(storage & Base::FP8_MANTISSA_MASK);
    }


    friend bool isnan(float_e5m2_t const& x) {
      return x.storage == uint8_t(0x7f);
    }

};

int main() {
    // float_e5m2_t half;
    // for (int i = 0; i < 256; i++) {
    //     half.storage = uint8_t(i);
    //     float float_val = float_e5m2_t::to_float(half);
    //     uint32_t* float_bits = reinterpret_cast<uint32_t*>(&float_val);
    //     printf("0x%x, 0x%x, %f\n", half.storage, *float_bits, float_val);
    //     // printf("0x%x, ", *float_bits);
    // }
    std::vector<uint8_t> x = {123, 127, 127, 251, 127, 127, 127};
    // std::vector<uint8_t> x = {124, 125, 126, 252, 253, 254, 255};
    for (auto& i : x) {
        float_e5m2_t half;
        half.storage = i;
        float float_val = float_e5m2_t::to_float(half);
        printf("%f\n", float_val);
    }
    return 0;
}
