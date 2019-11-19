#include <cstddef>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>             //  Included by default for CUDA code

#include <thrust/transform.h>         //  contains thrust::transform
#include <thrust/execution_policy.h>  //  contains thurst::device

//-----------------------------------------------------------------------------
//  Macros

//  Making functions available on device
#define DEVICE_CALLABLE __host__ __device__

//  Simple error check
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
//  Unroll function template

namespace detail {
  template< typename F, std::size_t... Is >
  constexpr void unroll_impl( F fun, std::index_sequence< Is... > ) {
    ( fun( std::integral_constant<std::size_t, Is>{} ), ... );
  }
}

//  Arbitrary lambda unroller
template< std::size_t N, typename F >
constexpr void unroll( F fun ) {
  detail::unroll_impl( fun, std::make_index_sequence<N>{} );
}

//-----------------------------------------------------------------------------
//  Constants

//  Max grid size (arbitrary, adjusted for older CUDA devices)
std::size_t constexpr max_grid_size = 65535;

//  Number of threads per block in the grid
std::size_t constexpr block_size    = 512;

//  Unroll factor
std::size_t constexpr unroll_n      = 4;

//  Number of elements processed per block
std::size_t constexpr elmt_per_block = block_size * unroll_n;

//-----------------------------------------------------------------------------
//  Transform

namespace detail {
  //  Transform CUDA kernel
  template< std::size_t U = unroll_n, typename InputIt, typename OutputIt, typename F >
  __global__ void transform_ker( InputIt in, OutputIt out, F fun )
  {
    auto const index = blockIdx.x * block_size + threadIdx.x;
    auto const thread_number = block_size * gridDim.x;

    //  Unrolled lambda
    unroll<U>( [&] ( auto I ) {
      out[ index ] = fun( in[ index + ( I * thread_number ) ] );
    } );
  }
}

//  Transform kernel wrapper (runs on CPU, calls the kernel declared above)
template< typename InputIt, typename OutputIt, typename F >
void transform( InputIt in_begin, InputIt in_end, OutputIt out_begin, F fun )
{
  using detail::transform_ker;

  while( in_begin < in_end
      && std::size_t( in_end - in_begin ) >= elmt_per_block )
  {
    //  Number of elements to process
    auto const remainer         = in_end - in_begin;

    //  Number of blocks required
    auto const remainer_blocks  = remainer / (block_size * unroll_n);
    auto const grid_size        = remainer_blocks < max_grid_size
                                    ? remainer_blocks
                                    : max_grid_size;

    //  Calling the kernel to process as many elements as possible
    transform_ker <<< grid_size, block_size >>> ( in_begin, out_begin, fun );

    in_begin  += grid_size * elmt_per_block;
    out_begin += grid_size * elmt_per_block;
  }

  //
  if( in_begin < in_end )
    transform_ker<1> <<< 1, in_end - in_begin >>> ( in_begin, out_begin, fun );

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

  vec_t a( 8192, 2.f ), b( 8192 );

  //  Homemade kernel
  transform( a.data(), a.data() + a.size(), b.data(),
    [] DEVICE_CALLABLE ( auto v ) { return v + 2; } );

  //  Thrust kernel
  thrust::transform( thrust::device, a.data(), a.data() + a.size(), b.data(),
    [] DEVICE_CALLABLE ( auto v ) { return v + 2; } );

  for( auto& elmt : b ) std::cout << elmt << '\n';

  std::cout << "end\n";

  return 0;
}
