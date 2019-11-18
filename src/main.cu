#include <cstddef>
#include <iostream>
#include <sstream>
#include <vector>

//-----------------------------------------------------------------------------
//  Macros

#define DEVICE_CALLABLE __host__ __device__

#define CUDA_ERROR_CHECK                                                      \
if( auto err = cudaGetLastError(); err != cudaSuccess )                       \
{                                                                             \
   std::stringstream ss;                                                      \
   ss << "CUDA Runtime error at: " << __FILE__ << ':' << __LINE__             \
      << ", in function: " << __func__ << ". CUDA error string: "             \
      << cudaGetErrorString( err );                                           \
   throw std::runtime_error( ss.str() );                                      \
}

//-----------------------------------------------------------------------------
//  Constants

std::size_t constexpr max_grid_size = 65535;
std::size_t constexpr block_size    = 512;

//-----------------------------------------------------------------------------
//  Transform

template< typename InputIt, typename OutputIt, typename F >
__global__ void transform_ker( InputIt in, OutputIt out, F fun )
{
  auto const index = blockIdx.x * block_size + threadIdx.x;
  out[ index ] = fun( in[ index ] );
}

template< typename InputIt, typename OutputIt, typename F >
void transform( InputIt in_begin, InputIt in_end, OutputIt out_begin, F fun )
{
  while( in_begin < in_end && std::size_t( in_end - in_begin ) >= block_size )
  {
    auto const remainer         = in_end - in_begin;
    auto const remainer_blocks  = remainer / block_size;
    auto const grid_size        = remainer_blocks < max_grid_size
                                    ? remainer_blocks
                                    : max_grid_size;

    transform_ker <<< grid_size, block_size >>> ( in_begin, out_begin, fun );

    in_begin  += grid_size * block_size;
    out_begin += grid_size * block_size;
  }

  if( in_begin < in_end )
    transform_ker <<< 1, in_end - in_begin >>> ( in_begin, out_begin, fun );

  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK;
}

//-----------------------------------------------------------------------------
//  CUDA managed memory allocator

template <typename T>
struct cuda_managed_alloc
{
  using value_type      = T;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;

  static T* allocate( std::size_t n ) {
    T* res;
    cudaMallocManaged( (void**)&res, n * sizeof(T) );
    return res;
  }

  static void deallocate( T* ptr, std::size_t ) {
    cudaFree( ptr );
  }
};

template<typename T>
using cuda_vector = std::vector<T, cuda_managed_alloc<T>>;

//----------------------------------------------------------------------------
//  Main

int main(int, char const *[])
{
  using vec_t = std::vector< float, cuda_managed_alloc<float> >;

  vec_t a( 8192, 0.f ), b( 8192, 0.f );

  transform( a.data(), a.data() + a.size(), b.data(),
    [] DEVICE_CALLABLE ( auto ) { return 2; } );

  for( auto& elmt : b ) std::cout << elmt << '\n';

  std::cout << "end\n";

  return 0;
}
