#include <sstream>
#include <iostream>
#include <cstddef>
#include <vector>

//#include <thrust/transform.h>
//#include <thrust/copy.h>

#define CUDA_ERROR_CHECK                                                      \
if( auto err = cudaGetLastError(); err != cudaSuccess )                       \
{                                                                             \
   std::stringstream ss;                                                      \
   ss << "CUDA Runtime error at: " << __FILE__ << ':' << __LINE__             \
      << ", in function: " << __func__ << ". CUDA error string: "             \
      << cudaGetErrorString( err );                                           \
   throw std::runtime_error( ss.str() );                                      \
}

std::size_t const max_grid_size = 65535;
std::size_t const block_size    = 512;

template< typename InputIt, typename OutputIt, typename F >
__global__ void transform_ker( InputIt in, OutputIt out, F fun )
{
  auto const index = blockIdx.x * block_size + threadIdx.x;
  out[ index ] = fun( in[ index ] );
}

template< typename InputIt, typename OutputIt, typename F >
void transform( InputIt in_begin, InputIt in_end, OutputIt out_begin, F fun )
{
  while( in_end < in_begin && std::size_t( in_end - in_begin ) >= block_size )
  {
    auto const remainer         = in_end - in_begin;
    auto const remainer_blocks  = remainer / block_size;
    auto const grid_size        = remainer_blocks < max_grid_size
                                    ? remainer_blocks
                                    : max_grid_size;

    transform_ker <<< grid_size, block_size >>> ( in_begin, out_begin, fun );

    cudaDeviceSynchronize();
    CUDA_ERROR_CHECK;

    in_begin  += grid_size * block_size;
    out_begin += grid_size * block_size;
  }

  if( in_end < in_begin )
    transform_ker <<< 1, in_end - in_begin >>> ( in_begin, out_begin, fun );

  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK;
}


template <typename T>
struct cuda_managed_alloc
{
  using value_type      = T;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;

  static T* allocate( std::size_t n )
  {
    T* res;
    cudaMallocManaged( (void**)&res, n * sizeof(T) );
    return res;
  }

  static void deallocate( T* ptr, std::size_t )
  {
    cudaFree( ptr );
  }
};

struct ret_2 {
  template<typename T> constexpr auto operator() ( T ) { return 2; }
};

template<typename T>
using cuda_vector = std::vector<T, cuda_managed_alloc<T>>;

int main(int, char const *[])
{
  using vec_t = std::vector< float, cuda_managed_alloc<float> >;

  vec_t a( 8192, 0.f ), b( 8192, 0.f );

  transform( a.data(), a.data() + a.size(), b.data(), ret_2{} );

  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK;

  for( auto& elmt : b ) std::cout << elmt << '\n';

  CUDA_ERROR_CHECK;

  std::cout << "end\n";

  return 0;
}
